import os
import wandb
import numpy as np

yaml = ['../yaml/cub_5shot.yaml']

api = wandb.Api()
runs = api.runs("william-yang/BeyondObjects_Sweep")

jobs = []

for y in yaml:
    for model in ['clip', 'imagenet']:
        accs = []
        configs = []
        for info in runs:
            if info.config['yaml_file'] == y and info.config['model_type'] == model and info.state == "finished":
                accs.append(info.summary['val/best_top1'])
                configs.append((info.config['lr'], info.config['wd'], info.config['lambda_1']))
        if model == 'imagenet':
            assert len(accs) == 18
        elif model == 'clip':
            assert len(accs) == 16
        
        best_config = configs[np.argmax(accs)]
        jobs.append((model, y, best_config[0], best_config[1], best_config[2]))
        print(model, y, best_config)

for job in jobs:
    os.system(f"sbatch run_final.sh {job[0]} {job[1]} {job[2]} {job[3]} {job[4]}")
            