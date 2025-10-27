
import sys
import logging
import random
import atexit
import getpass
import shutil
import time
import os
import yaml
import json
import argparse
from os.path import join as ospj

_MODEL_TYPE = ("clip", "imagenet")


class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self.log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def str2bool(v):
    if v == "":
        return None
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2none(v):
    if v is None:
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return v

def int2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return int(v)

def float2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return float(v)

def list_int2none(vs):
    return_vs = []
    for v in vs:
        if v is None:
            pass
        elif v.lower() in ('none', 'null'):
            v = None
        else:
            v = int(v)
        return_vs.append(v)
    return return_vs

def set_local(args):
    assert args.yaml_file is not None, "yaml_file is required"
    yaml_file = args.yaml_file 
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    return args_local

def get_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_type', type=str2none, default=None,
                        choices=_MODEL_TYPE)
    parser.add_argument('--clip_version', type=str, default='ViT-B/16')
    parser.add_argument('--clip_download_dir', type=str, default='clip_models/')

    # CLIP setting
    parser.add_argument("--is_lora_image", type=str2bool, default=True)
    parser.add_argument("--is_lora_text", type=str2bool, default=True)

    # Data
    parser.add_argument("--n_img_per_cls", type=int2none, default=100)
    parser.add_argument("--is_mix_aug", type=str2bool, default=False,
                        help="use mixup and cutmix")
    parser.add_argument("--is_pooled_fewshot", type=str2bool, default=False)
    parser.add_argument("--lambda_1", type=float2none, default=0,
                        help="weight for loss from real/synth data")

    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=str2bool,
        default=True,
        help="Whether or not to use mixed precision for training.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU. Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--batch_size_eval",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--wd",
        type=float2none,
        default=1e-4,
        help="Weight decay for the SGD optimizer.",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float2none,
        help="Maximum learning rate at the end of linear warmup.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=25,
        type=int,
        help="Number of training epochs for the learning-rate-warm-up phase.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at the end of training.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckpt_freq",
        default=100,
        type=int,
        help="Frequency of intermediate checkpointing.",
    )
    parser.add_argument(
        "--seed",
        default=22,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument("--yaml_file", type=str, default=None)
    # wandb args
    parser.add_argument('--log', type=str, default='wandb', help='How to log')
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--final_run', action='store_true', help='Whether to run the final run after the sweep')

    args = parser.parse_args()
    args_local = set_local(args)
    args.real_data_dir = args_local["real_data_path"]
    args.fewshot_data_path = args_local["data_path"]
    args.synth_train_data_dir = args_local["synthetic_path"]
    args.classnames = args_local["classnames"]
    args.dataset = args_local["dataset"]
    if "real_data_multiplier" in args_local:
        args.real_data_multiplier = args_local["real_data_multiplier"]
    else:
        args.real_data_multiplier = None
    return args


