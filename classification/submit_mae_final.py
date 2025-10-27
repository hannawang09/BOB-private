import os
import wandb
import numpy as np

yaml = ['../yaml/aircraft_5shot.yaml']

api = wandb.Api()
runs = api.runs("william-yang/BeyondObjects_MAE_Sweep")

jobs = []

for y in yaml:
    accs = []
    configs = []
    for info in runs:
        if info.config['yaml_file'] == y and info.state == "finished":
            accs.append(info.summary['Best ID Accuracy'])
            configs.append((info.config['blr'], info.config['layer_decay'], info.config['lambda_1']))
    assert len(accs) == 8, (y, accs)
    best_config = configs[np.argmax(accs)]
    jobs.append((best_config[0], best_config[1], best_config[2], y))

for job in jobs:
    os.system(f"sbatch run_mae_final.sh {job[0]} {job[1]} {job[2]} {job[3]}")
        