#!/bin/sh
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8    # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=32G             # memory 
#SBATCH -o run.txt         # send stdout to outfile
#SBATCH -e run.txt         # send stderr to errfile
#SBATCH -t 12:00:00           # time requested in hour:minute:second
#SBATCH -x node401,node002,node402,node017
####SBATCH --begin=now+4hour


OUTPUT_DIR="outputs"

NIPC=100
LR=$3
MIN_LR=1e-8
WD=$4
EPOCH=100
WARMUP_EPOCH=3
IS_MIX_AUG=True

IS_POOLED=True
LAMBDA_1=$5


#WANDB_MODE=disabled \
python main.py \
--model_type=$1 \
--output_dir=$OUTPUT_DIR \
--is_lora_image=True \
--is_lora_text=True \
--is_pooled_fewshot=$IS_POOLED \
--lambda_1=$LAMBDA_1 \
--epochs=$EPOCH \
--warmup_epochs=$WARMUP_EPOCH \
--log=wandb \
--lr=$LR \
--wd=$WD \
--min_lr=$MIN_LR \
--is_mix_aug=$IS_MIX_AUG \
--wandb_project=BeyondObjects_Final \
--yaml_file=$2 \
--final_run 
