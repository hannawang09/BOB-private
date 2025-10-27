

import os
import sys
import json
import time
import math
import random
import datetime
import traceback
from pathlib import Path
from os.path import join as ospj
import wandb
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision.transforms import v2
import torchvision

from utils import (
    fix_random_seeds,
    cosine_scheduler,
    MetricLogger,
)

from config import get_args
from data import validation_data_loader, get_synth_train_data_loader, get_data_loader
from models.clip import CLIP
import numpy as np



def load_val_data_loader(args):
    test_loader = validation_data_loader(
        real_train_data_dir=args.real_data_dir,
        dataset=args.dataset, 
        batch_size=args.batch_size_eval,
        model_type=args.model_type,
    )
    return test_loader

def load_data_loader(args):
    test_loader = get_data_loader(
        real_test_data_dir=args.real_data_dir,
        dataset=args.dataset, 
        batch_size=args.batch_size_eval,
        model_type=args.model_type,
    )
    return test_loader


def load_synth_train_data_loader(args):
    synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=args.synth_train_data_dir,
        batch_size=args.batch_size,
        n_img_per_cls=args.n_img_per_cls,
        real_train_fewshot_data_dir=args.fewshot_data_path,
        is_pooled_fewshot=args.is_pooled_fewshot,
        model_type=args.model_type,
        classnames=args.classnames,
        is_rand_aug=True,
        real_data_multiplier=args.real_data_multiplier,
    )
    return synth_train_loader


def main(args):
    args.n_classes = len(args.classnames)

    if args.final_run and 'imbalanced' in args.dataset: 
        f = open(args.fewshot_data_path.replace('train', 'train.txt'),'r')
        lines = f.readlines()
        f.close()

        counts = {}
        for line in lines:
            label = line.split('/')[-2]
            if label not in counts:
                counts[label]=1
            else:
                counts[label]+=1
    else:
        counts = None

    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    # ==================================================
    # Data loader
    # ==================================================
    val_loader = load_val_data_loader(args)
    train_loader = load_synth_train_data_loader(args)
    if args.final_run:
        test_loader = load_data_loader(args)

    # ==================================================
    # Model and optimizer
    # ==================================================
    if args.model_type == "clip":
        model = CLIP(
            dataset=args.dataset,
            is_lora_image=args.is_lora_image,
            is_lora_text=args.is_lora_text,
            clip_download_dir=args.clip_download_dir,
            clip_version=args.clip_version,
            classnames=args.classnames,
        )
        params_groups = model.learnable_params()
    elif args.model_type == 'imagenet':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, args.n_classes)
        params_groups = model.parameters()

    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # CutMix and MixUp augmentation
    if args.is_mix_aug:
        cutmix = v2.CutMix(num_classes=args.n_classes)
        mixup = v2.MixUp(num_classes=args.n_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = None

    scheduler = None
    optimizer = torch.optim.AdamW(
        params_groups, lr=args.lr, weight_decay=args.wd,
    )
    args.lr_schedule = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.min_lr,
    )

    fp16_scaler = None
    if args.use_fp16:
        # mixed precision training
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ==================================================
    # Loading previous checkpoint & initializing tensorboard
    # ==================================================

    if args.log == 'wandb':
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        # _ = os.system('wandb login {}'.format(args.wandb_key))
        # os.environ['WANDB_API_KEY'] = args.wandb_key
        wandb.init(
            project=args.wandb_project,
            settings=wandb.Settings(start_method='fork'),
            config=vars(args)
        )
        args.wandb_url = wandb.run.get_url()

    # ==================================================
    # Training
    # ==================================================
    print("=> Training starts ...")
    start_time = time.time()

    best_stats = {}
    best_stats_val = {}
    best_top1 = 0.
    best_top1_val = 0.
    best_top1_val_test = {}

    for epoch in range(0, args.epochs):
        train_stats, best_stats, best_top1 = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
            val_loader, best_stats, best_top1, 
        )

        val_stats = eval(
            model, criterion, val_loader, epoch, fp16_scaler, args, "val", None)

        if args.final_run:
            test_stats = eval(
                model, criterion, test_loader, epoch, fp16_scaler, args, "test", counts)
            if test_stats["test/top1"] > best_top1:
                best_top1 = test_stats["test/top1"]
                best_stats = test_stats

        if counts is not None:
            if val_stats["val/top1"] > best_top1_val:
                best_top1_val = val_stats["val/top1"]
                best_stats_val = val_stats
                if args.final_run:
                    best_top1_val_test = test_stats

        if epoch + 1 == args.epochs:
            val_stats['val/best_top1'] = best_top1_val
            val_stats['val/best_loss'] = best_stats_val["val/loss"]
            if args.final_run:
                test_stats['test/best_top1'] = best_stats["test/top1"]
                test_stats['test/best_loss'] = best_stats["test/loss"]
                if counts is None:
                    test_stats['test/best_top1_fromval'] = best_top1_val_test["test/top1"]
                else:
                    test_stats['test/best_top1_fromval'] = best_top1_val_test["test/expected_acc_per_class"]
                    test_stats['test/best_many_acc'] = best_top1_val_test["test/many_acc"]
                    test_stats['test/best_medium_acc'] = best_top1_val_test["test/medium_acc"]
                    test_stats['test/best_few_acc'] = best_top1_val_test["test/few_acc"]

        if args.log == 'wandb':
            train_stats.update({"epoch": epoch})
            wandb.log(train_stats)
            wandb.log(val_stats)    
            if args.final_run:
                wandb.log(test_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    model, criterion, data_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
    val_loader, best_stats, best_top1,
):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()

    for it, batch in enumerate(
        metric_logger.log_every(tqdm(data_loader), 100, header)
    ):
        if args.is_pooled_fewshot:
            image, label, is_real = batch
        else:
            image, label = batch

        label_origin = label
        label_origin = label_origin.cuda(non_blocking=True)

        # apply CutMix and MixUp augmentation
        if args.is_mix_aug:
            p = random.random()
            if p >= 0.2:
                pass
            else:
                if args.is_pooled_fewshot:
                    new_image = torch.zeros_like(image)
                    new_label = torch.stack([torch.zeros_like(label)] * args.n_classes, dim=1).mul(1.0)

                    image_real, label_real = image[is_real==1], label[is_real==1]
                    image_synth, label_synth = image[is_real==0], label[is_real==0]

                    image_real, label_real = cutmix_or_mixup(image_real, label_real)
                    image_synth, label_synth = cutmix_or_mixup(image_synth, label_synth)

                    new_image[is_real==1] = image_real
                    new_image[is_real==0] = image_synth
                    new_label[is_real==1] = label_real
                    new_label[is_real==0] = label_synth

                    image = new_image
                    label = new_label

                else:
                    image, label = cutmix_or_mixup(image, label)
            

        it = len(data_loader) * epoch + it  # global training iteration

        image = image.squeeze(1).to(torch.float16).cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = args.wd

        # forward pass
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            logit = model(image)
            if args.is_pooled_fewshot:
                loss_real = criterion(logit[is_real == 1], label[is_real == 1])
                loss_synth = criterion(logit[is_real == 0], label[is_real == 0])
                loss = 0
                if not torch.isnan(loss_real):
                    loss += args.lambda_1 * loss_real
                if not torch.isnan(loss_synth):
                    loss += (1 - args.lambda_1) * loss_synth
            else:
                loss = criterion(logit, label)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # parameter update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        with torch.no_grad():
            acc1, acc5 = get_accuracy(logit.detach(), label_origin, topk=(1, 5))
            metric_logger.update(top1=acc1.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if scheduler is not None:
            scheduler.step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)

    return {"train/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}, best_stats, best_top1


@torch.no_grad()
def eval(model, criterion, data_loader, epoch, fp16_scaler, args, prefix, counts):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    targets = []
    outputs = []

    model.eval()

    for it, (image, label) in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):

        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if args.model_type == "imagenet":
                output = model(image)
            else:
                output = model(image, phase="eval")
            loss = criterion(output, label)

        acc1, acc5 = get_accuracy(output, label, topk=(1, 5))

        # record logs
        metric_logger.update(loss=loss.item())
        metric_logger.update(top1=acc1.item())
        metric_logger.update(top5=acc5.item())

        targets.append(label)
        outputs.append(output)

    metric_logger.synchronize_between_processes()
    print("Averaged test stats:", metric_logger)

    stat_dict = {"{}/{}".format(prefix, k): meter.global_avg for k, meter in metric_logger.meters.items()}
    targets = torch.hstack(targets)
    outputs = torch.vstack(outputs)

    if counts is not None:
        n_classes = len(args.classnames)
        acc_per_class = [
            get_accuracy(outputs[targets == cls_idx], targets[targets == cls_idx], topk=(1,))[0].item() 
            for cls_idx in range(n_classes)
        ]
        acc_per_class = np.array(acc_per_class)
        stat_dict.update({"{}/expected_acc_per_class".format(prefix): np.mean(acc_per_class)})
        
        all_classes = np.arange(n_classes)
        if 'flower' in args.dataset:
            many_classes = np.array([cls_idx for cls_idx in all_classes if counts[args.classnames[cls_idx]] > 30])
            medium_classes = np.array([cls_idx for cls_idx in all_classes if counts[args.classnames[cls_idx]] >= 10 and counts[args.classnames[cls_idx]] <= 30])
            few_classes = np.array([cls_idx for cls_idx in all_classes if counts[args.classnames[cls_idx]] < 10])
        elif 'cub' in args.dataset:   
            many_classes = np.array([cls_idx for cls_idx in all_classes if counts[args.classnames[cls_idx]] > 20])
            medium_classes = np.array([cls_idx for cls_idx in all_classes if counts[args.classnames[cls_idx]] >= 5 and counts[args.classnames[cls_idx]] <= 20])
            few_classes = np.array([cls_idx for cls_idx in all_classes if counts[args.classnames[cls_idx]] < 5])
        else:
            assert False, "imbalanced dataset not supported"

        for cls_idx, acc in enumerate(acc_per_class):
            stat_dict[prefix + "/" + args.classnames[cls_idx] + '_cls-acc'] = acc

        many_acc = np.mean(acc_per_class[many_classes])
        medium_acc = np.mean(acc_per_class[medium_classes])
        few_acc = np.mean(acc_per_class[few_classes])
        stat_dict.update({"{}/many_acc".format(prefix): many_acc})
        stat_dict.update({"{}/medium_acc".format(prefix): medium_acc})
        stat_dict.update({"{}/few_acc".format(prefix): few_acc})

    return stat_dict


def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]



def save_model(args, model, optimizer, epoch, fp16_scaler, file_name):
    state_dict = model.state_dict()
    save_dict = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch + 1,
        "args": args,
    }
    if fp16_scaler is not None:
        save_dict["fp16_scaler"] = fp16_scaler.state_dict()
    torch.save(save_dict, os.path.join(args.output_dir, file_name))


if __name__ == "__main__":
    args = get_args()
    main(args)
