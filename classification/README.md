# Instructions for classification

After fine-tuning the T2I model and creating our synthetic dataset, we train a downstream classifier. 

## Training and evaluating on validation set
To fine-tune a CLIP model

`bash run_validation.sh clip [lr] [weight decay] [lambda] [yaml file]`

To fine-tune a ImageNet pre-trained ResNet-50

`bash run_validation.sh imagenet [lr] [weight decay] [lambda] [yaml file]`

To fine-tune a ImageNet pre-trained MAE

`bash run_mae_validation.sh [lr] [weight decay] [lambda] [yaml file]`

To automate the hyperparameter sweep across multiple GPUs

`python hyperparameter_sweep.py`

## Training and evaluating on test set
After sweeping the hyperparameters and logging them to wandb, modify `submit_final.py` and `submit_mae_final.py` to launch the final runs.
