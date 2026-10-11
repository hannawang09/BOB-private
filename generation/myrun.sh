python generate.py \
--yaml_file ../yaml/plantdoc_8shot.yaml \
--batch_size=1 \
--guidance_scale=2.0 \
--num_inference_steps=50 \
--n_set_split=1 \
--split_idx=0 \
--seed=42 \
> output.txt 2>&1


python generate.py \
--yaml_file ../yaml/plantdoc_16shot.yaml \
--batch_size=1 \
--guidance_scale=2.0 \
--num_inference_steps=50 \
--n_set_split=1 \
--split_idx=0 \
--seed=42 \
> output1.txt 2>&1