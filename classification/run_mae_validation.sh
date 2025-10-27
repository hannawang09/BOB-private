#!/bin/sh
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4    # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=16G             # memory 
#SBATCH -o run.txt         # send stdout to outfile
#SBATCH -e run.txt         # send stderr to errfile
#SBATCH -t 12:00:00           # time requested in hour:minute:second
#####SBATCH --begin=now+20minutes 
#SBATCH -x node028,node017

#WANDB_MODE=disabled \
python main_mae.py \
    --accum_iter 1 \
    --batch_size 64 \
    --model vit_base_patch16 \
    --finetune /n/fs/wy-project/mae_pretrain_vit_base.pth \
    --epochs 10 \
    --blr $1 --layer_decay $2 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --yaml_file $4 \
    --lambda_1 $3
   
