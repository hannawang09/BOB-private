import random
from os.path import join as ospj

import torch
import yaml

from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import numpy as np
import os

import argparse




def set_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dirs(fpath_dir):
    if not os.path.exists(fpath_dir):
        os.makedirs(fpath_dir)

def get_pipe(model, lora_path, device, is_tqdm):
    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=not is_tqdm)
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
    return pipe

def get_prompt_embeds(pipe, prompts, device):
    text_inputs = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    if (
        hasattr(pipe.text_encoder.config, "use_attention_mask")
        and pipe.text_encoder.config.use_attention_mask
    ):
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = pipe.text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


class GenerateImage:
    def __init__(
        self,
        pipe,
        device,
        guidance_scale,
        num_inference_steps,
        n_img_per_class,
        save_dir,
        bs,
        caption_path,
        template,
        real_data_multiplier,
        counts,
    ):
        self.pipe = pipe
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.n_img_per_class = n_img_per_class
        self.save_dir = save_dir
        self.bs = bs
        self.template = template
        self.real_data_multiplier = real_data_multiplier
        self.counts = counts
        background_captions = []
        with open(f'{caption_path}/background.tsv', 'r') as f:
            next(f)
            for line in f:
                line = line.strip().split('\t')
                background_captions.append(line[1].rstrip('.'))
        pose_captions = []
        with open(f'{caption_path}/pose.tsv', 'r') as f:
            next(f)
            for line in f:
                line = line.strip().split('\t')
                pose_captions.append(line[1].rstrip('.'))
        assert len(background_captions) == len(pose_captions), "Background and pose captions must have the same length"
        self.background_captions = background_captions
        self.pose_captions = pose_captions

    def save_data(self, outputs, save_dir, count):
        images = outputs.images
        for image in images:
            fpath = ospj(save_dir, f"{count}.png")
            image = image.resize((512, 512))
            image.save(fpath)
            count += 1
        return count

    def run_pipe(self, prompts):
        if isinstance(prompts, list):
            prompt_embeds = None
        elif isinstance(prompts, torch.Tensor):
            prompt_embeds = prompts
            prompts = None

        lora_scale = 1

        outputs = self.pipe(
            prompt=prompts,
            prompt_embeds=prompt_embeds,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            cross_attention_kwargs={"scale": lora_scale},
        )
        return outputs

    def set_save_dir(self, classname):
        save_dir = ospj(self.save_dir, classname)

        make_dirs(save_dir)

        return save_dir

    def run(self, classname):
        save_dir = self.set_save_dir(classname)

        count = 0
        for _ in range(200 - self.counts[classname]*self.real_data_multiplier):
            caption_index = random.randint(0, len(self.background_captions) - 1)
            background = self.background_captions[caption_index]
            pose = self.pose_captions[caption_index]
            prompt = self.template.format(classname, background, pose)
            with open(ospj(save_dir, "prompts.txt"), "a") as f:
                f.write(prompt + "\n")
            # generate images
            outputs = self.run_pipe([prompt])
            # save
            count = self.save_data(outputs, save_dir, count)


def set_local(dataset):
    yaml_file = "local_caption.yaml"
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    return args_local


def main(args):
    print(f"seed: {args.seed}")
    f = open(args.data_path.replace('train', 'train.txt'),'r')
    lines = f.readlines()
    f.close()

    counts = {}
    for line in lines:
        label = line.split('/')[-2]
        if label not in counts:
            counts[label]=1
        else:
            counts[label]+=1

    # set local arguments
    model_dir = args.pretrained_model_name_or_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load SD pipeline
    pipe = get_pipe(args.pretrained_model_name_or_path, args.lora_path, device, args.is_tqdm)

    # load instance
    generate_image = GenerateImage(
        pipe=pipe,
        device=device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        n_img_per_class=args.n_img_per_class,
        save_dir=args.save_dir,
        bs=args.batch_size,
        caption_path=args.caption_path,
        template=args.template,
        real_data_multiplier=args.real_data_multiplier,
        counts=counts,
    )

    iters = args.classnames

    # parallel computing
    step = len(iters) // args.n_set_split
    start_idx = args.split_idx * step
    end_idx = (args.split_idx + 1) * step if (args.split_idx + 1) != args.n_set_split else len(iters)
    print(
        f"SPLIT!! Out of {len(args.classnames)} pairs, we generate from idx {start_idx} to {end_idx}."
    )
    iters_partial = iters[start_idx:end_idx]

    # generate & save synthetic images
    for classname in tqdm(iters_partial, total=len(iters_partial)):
        # run
        set_seed(args.seed)
        generate_image.run(classname)

def set_local(args):
    yaml_file = args.yaml_file
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    return args_local


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image generation script.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--n_set_split", type=int, required=True)
    parser.add_argument("--split_idx", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--is_tqdm", type=bool, default=True)
    parser.add_argument("--yaml_file", type=str, required=True)
    args = parser.parse_args()
    args_local = set_local(args)
    args.pretrained_model_name_or_path = args_local["model"]
    args.lora_path = args_local["lora_path"]
    args.caption_path = args_local["caption_path"]
    args.template = args_local["template"]
    args.classnames = args_local["classnames"]
    args.save_dir = args_local["synthetic_path"]
    args.n_img_per_class = args_local["num_synth"]
    args.real_data_multiplier = args_local["real_data_multiplier"]
    args.data_path = args_local["data_path"]
    main(args)
