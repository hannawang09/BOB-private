#!/bin/sh
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=1    # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:1          # the number of GPUs requested
#SBATCH --mem=4G             # memory 
#SBATCH -o run.txt         # send stdout to outfile
#SBATCH -e run.txt         # send stderr to errfile
#SBATCH -t 6:00:00           # time requested in hour:minute:second
#SBATCH -x node401,node402,node017,node002

N_SET_SPLIT=50
SPLIT_IDX="$1"

python generate.py \
--yaml_file ../yaml/aircraft_5shot.yaml \
--batch_size=1 \
--guidance_scale=2.0 \
--num_inference_steps=50 \
--n_set_split=$N_SET_SPLIT \
--split_idx=$SPLIT_IDX \
--seed=42 \


