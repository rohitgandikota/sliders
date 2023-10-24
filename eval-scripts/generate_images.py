import torch
from PIL import Image
import argparse
import os, json, random
import pandas as pd
import matplotlib.pyplot as plt
import glob, re,sys
from tqdm.auto import tqdm

from safetensors.torch import load_file
import matplotlib.image as mpimg
import copy
import gc
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor
from typing import Any, Dict, List, Optional, Tuple, Union
sys.path.insert(1, os.getcwd())
from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV


def flush():
    torch.cuda.empty_cache()
    gc.collect()
flush()




def sorted_nicely( l ):
    convert = lambda text: float(text) if text.replace('-','').replace('.','').isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('(-?[0-9]+.?[0-9]+?)', key) ]
    return sorted(l, key = alphanum_key)

def flush():
    torch.cuda.empty_cache()
    gc.collect()
    
def generate_images(model_name, prompts_path, save_path, negative_prompt, device, guidance_scale , image_size, ddim_steps, num_samples,from_case, till_case, base, rank, start_noise):
    # Load scheduler, tokenizer and models.
    scales = [-2, -1, -.5, 0, .5, 1, 2]
    revision = None
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    weight_dtype = torch.float32

    # Load scheduler, tokenizer and models.
    noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.


    # Move unet, vae and text_encoder to device and cast to weight_dtype

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    name = os.path.basename(model_name)
    alpha = 4
    train_method = 'xattn'
    n = model_name.split('/')[-2]
    if 'noxattn' in n:
        train_method = 'noxattn'
    if 'hspace' in n:
        train_method+='-hspace'
        scales = [-5, -2, -1, 0, 1, 2, 5]
    if 'last' in n:
        train_method+='-last'
        scales = [-5, -2, -1, 0, 1, 2, 5]
    network_type = "c3lier"
    if train_method == 'xattn':
        network_type = 'lierla'

    modules = DEFAULT_TARGET_REPLACE
    if network_type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    network = LoRANetwork(
            unet,
            rank=rank,
            multiplier=1.0,
            alpha=alpha,
            train_method=train_method,
        ).to(device, dtype=weight_dtype)
    
    network.load_state_dict(torch.load(model_name))

    df = pd.read_csv(prompts_path)
    prompts = df.prompt
    seeds = df.evaluation_seed
    case_numbers = df.case_number

    folder_path = f'{save_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path+f'/all', exist_ok=True)
    scales_str = []
    for scale in scales:
        scale_str = f'{scale}'
        scale_str = scale_str.replace('0.5','half')
        scales_str.append(scale_str)
        os.makedirs(folder_path+f'/{scale_str}', exist_ok=True)
    height = image_size                        # default height of Stable Diffusion
    width = image_size                         # default width of Stable Diffusion

    num_inference_steps = ddim_steps           # Number of denoising steps

    guidance_scale = guidance_scale            # Scale for classifier-free guidance
    torch_device = device
    for _, row in df.iterrows():
        print(str(row.prompt),str(row.evaluation_seed))
        prompt = [str(row.prompt)]*num_samples
        batch_size = len(prompt)
        seed = row.evaluation_seed
        case_number = row.case_number
        if not (case_number>=from_case and case_number<=till_case):
            continue
        images_list = []
        for scale in scales:
            torch_device = device
            negative_prompt = None
            height = 512
            width = 512
            guidance_scale = 7.5

            generator = torch.manual_seed(seed) 
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

            max_length = text_input.input_ids.shape[-1]
            if negative_prompt is None:
                uncond_input = tokenizer(
                    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
            else:
                uncond_input = tokenizer(
                    [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            latents = torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(torch_device)

            noise_scheduler.set_timesteps(ddim_steps)

            latents = latents * noise_scheduler.init_noise_sigma

            latent_model_input = torch.cat([latents] * 2)
            for t in tqdm(noise_scheduler.timesteps):
                if t>start_noise:
                    network.set_lora_slider(scale=0)
                else:
                    network.set_lora_slider(scale=scale)
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                # predict the noise residual
                with network:
                    with torch.no_grad():
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            images_list.append(pil_images)
        for num in range(num_samples):
            fig, ax = plt.subplots(1, len(images_list), figsize=(4*(len(scales)),4))
            for i, a in enumerate(ax):
                images_list[i][num].save(f'{folder_path}/{scales_str[i]}/{case_number}_{num}.png')
                a.imshow(images_list[i][num])
                a.set_title(f"{scales[i]}",fontsize=15)
                a.axis('off')
            fig.savefig(f'{folder_path}/all/{case_number}_{num}.png',bbox_inches='tight')
            plt.close()
    del network, unet
    flush()
if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--negative_prompts', help='negative prompt', type=str, required=False, default=None)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--base', help='version of stable diffusion to use', type=str, required=False, default='1.4')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--till_case', help='continue generating from case_number', type=int, required=False, default=1000000)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=5)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--rank', help='rank of the LoRA', type=int, required=False, default=4)
    parser.add_argument('--start_noise', help='what time stamp to flip to edited model', type=int, required=False, default=650)
    
    args = parser.parse_args()
    
    model_name = args.model_name
    rank = args.rank
    if 'rank1' in model_name:
        rank = 1
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    till_case = args.till_case
    start_noise = args.start_noise
    base = args.base
    negative_prompts_path = args.negative_prompts
    if negative_prompts_path is not None:
        negative_prompt = ''
        with open(negative_prompts_path, 'r') as fp:
            vals = json.load(fp)
            for val in vals:
                negative_prompt+=val+' ,'
        print(f'Negative prompt is being used: {negative_prompt}')
    else:
        negative_prompt = None
    generate_images(model_name=model_name, prompts_path=prompts_path, save_path=save_path, negative_prompt=negative_prompt, device=device, guidance_scale = guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case, till_case=till_case, base=base, rank=rank, start_noise=start_noise)