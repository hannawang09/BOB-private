#!/bin/sh
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=96    # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:10          # the number of GPUs requested
#SBATCH --mem=500G             # memory 
#SBATCH -o run.txt         # send stdout to outfile
#SBATCH -e run.txt         # send stderr to errfile
#SBATCH -t 24:00:00           # time requested in hour:minute:second
#SBATCH -w node027


#WANDB_MODE=disabled \
accelerate launch finetune_t2i.py \
--yaml_file ../yaml/aircraft_5shot.yaml \
--resume_from_checkpoint=None \
--train_batch_size=8 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-4 \
--lr_scheduler="cosine" \
--lr_warmup_steps=100 \
--num_train_epochs=400 \
--is_tqdm=True \
--checkpointing_steps 50000 \
--checkpoints_total_limit 1 \
