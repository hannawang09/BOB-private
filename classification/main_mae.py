# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit

from engine_finetune import train_one_epoch, evaluate


from torch.utils.data import Dataset
from torchvision.models import ViT_L_16_Weights
from PIL import Image
import wandb

from torchvision.datasets.folder import default_loader
import torchvision
import pandas as pd

from data import get_synth_train_data_loader, get_data_loader, validation_data_loader
import yaml
from os.path import join as ospj
import torchvision.transforms.v2 as v2


def set_local(args):
    assert args.yaml_file is not None, "yaml_file is required"
    yaml_file = args.yaml_file 
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    return args_local

def str2bool(v):
    if v == "":
        return None
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument(
        "--batch_size_eval",
        default=64,
        type=int,
    )
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--is_pooled_fewshot", type=str2bool, default=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=10, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--ipc', type=int, default=-1)
    parser.add_argument('--distributed', action='store_true')

    parser.add_argument('--yaml_file', type=str, required=True)

    parser.add_argument('--final_run', action='store_true')

    parser.add_argument('--lambda_1', type=float, required=True)

    return parser

def set_local(args):
    assert args.yaml_file is not None, "yaml_file is required"
    yaml_file = args.yaml_file 
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    return args_local

def load_synth_train_data_loader(args):
    synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=args.synth_train_data_dir,
        batch_size=args.batch_size,
        n_img_per_cls=args.n_img_per_cls,
        real_train_fewshot_data_dir=args.fewshot_data_path,
        is_pooled_fewshot=args.is_pooled_fewshot,
        model_type='imagenet',
        classnames=args.classnames,
        is_rand_aug=True,
        real_data_multiplier=args.real_data_multiplier,
    )
    return synth_train_loader

def load_data_loader(args):
    test_loader = get_data_loader(
        real_test_data_dir=args.real_data_dir,
        dataset=args.dataset, 
        batch_size=args.batch_size_eval,
        model_type='imagenet',
    )
    return test_loader

def load_val_data_loader(args):
    test_loader = validation_data_loader(
        real_train_data_dir=args.real_data_dir,
        dataset=args.dataset, 
        batch_size=args.batch_size_eval,
        model_type='imagenet',
    )
    return test_loader



def main(args):
    args_local = set_local(args)
    args.real_data_dir = args_local["real_data_path"]
    args.fewshot_data_path = args_local["data_path"]
    args.synth_train_data_dir = args_local["synthetic_path"]
    args.classnames = args_local["classnames"]
    args.dataset = args_local["dataset"]
    args.n_img_per_cls = len(args.classnames)
    if "real_data_multiplier" in args_local:
        args.real_data_multiplier = args_local["real_data_multiplier"]
    else:
        args.real_data_multiplier = None
    if args.distributed:
        misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    cudnn.benchmark = True

    transforms = ViT_L_16_Weights.DEFAULT.transforms()

    data_loader_train = load_synth_train_data_loader(args)
    data_loader_val = load_val_data_loader(args)
    data_loader_test = load_data_loader(args)

    # Add the new configurations to the wandb.init call
    if args.final_run:
        wandb_title = "BeyondObjects_MAE_Final"
    else:
        wandb_title = "BeyondObjects_MAE_Sweep"
    if not args.distributed or args.local_rank == 0:
        wandb.init(
            project=wandb_title,
            config={
                'dataset': args.dataset,
                'yaml_file': args.yaml_file,
                'lambda_1': args.lambda_1,
                'blr': args.blr,
                'layer_decay': args.layer_decay,
                'wd': args.weight_decay,
                'drop_path': args.drop_path,
                'mixup': args.mixup,
                'cutmix': args.cutmix,
                'reprob': args.reprob,
            }
        )


    log_writer = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        cutmix = v2.CutMix(alpha=args.cutmix, num_classes=args.n_img_per_cls)
        mixup = v2.MixUp(alpha=args.mixup, num_classes=args.n_img_per_cls)
        mixup_fn = v2.RandomChoice([cutmix, mixup])
    
    model = models_vit.__dict__[args.model](
        num_classes=args.n_img_per_cls,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
     
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_id_accuracy = 0.0
    max_ood_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        val_stats = evaluate(data_loader_val, model, device)
        if args.final_run:
            test_stats = evaluate(data_loader_test, model, device)

        if args.final_run and val_stats["acc1"] > max_id_accuracy:
            best_test_accuracy = test_stats["acc1"]
        max_id_accuracy = max(max_id_accuracy, val_stats["acc1"])

        if not args.distributed or args.local_rank == 0:
            if args.final_run:
                wandb.log({"ID accuracy": val_stats["acc1"], "OOD accuracy": test_stats["acc1"],  "train loss": train_stats["loss"], "lr": train_stats["lr"],
                            "Best ID Accuracy": max_id_accuracy, "Best test Accuracy": best_test_accuracy})
            else:
                wandb.log({"ID accuracy": val_stats["acc1"],  "train loss": train_stats["loss"], "lr": train_stats["lr"],
                            "Best ID Accuracy": max_id_accuracy})


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
