import os

yaml_file = '../yaml/imb_flower_0.01.yaml'

# CLIP hyperparameters
for lr in [1e-4, 1e-5, 1e-6, 1e-7]:
    for wd in [5e-4, 1e-4]:
        for lambda_1 in [0.5, 0.8]:
            os.system(f"sbatch run_validation.sh clip {lr} {wd} {lambda_1} {yaml_file}")

# ImageNet hyperparameters
for lr in [1e-3, 1e-4, 1e-5]:
    for wd in [0.01, 1e-4, 0.0]:
        for lambda_1 in [0.5, 0.8]:
            os.system(f"sbatch run_validation.sh imagenet {lr} {wd} {lambda_1} {yaml_file}")

for lr in [1e-3, 5e-4]:
    for ld in [0.75, 0.65]:
        for l in [0.5, 0.8]:
            os.system(f'sbatch run_mae_validation.sh {lr} {ld} {l} {yaml_file}')
