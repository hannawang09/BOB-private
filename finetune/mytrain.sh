python finetune_t2i.py \
--yaml_file ../yaml/plantdoc_8shot.yaml \
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
> output.txt 2>&1


python finetune_t2i.py \
--yaml_file ../yaml/plantdoc_16shot.yaml \
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
> output1.txt 2>&1