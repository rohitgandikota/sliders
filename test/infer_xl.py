# ref:
# - https://github.com/huggingface/diffusers/blob/f74d5e1c2f7caa645e95ac296b6a252276e6d185/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L655-L821


import sys

sys.path.append("..")

import gc

import torch
import torchvision

from diffusers import AutoencoderKL

import train_util
import model_util
import config_util

SDXL_V09_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-0.9"
SDXL_VAE_FP16_FIX_MODEL_NAME = "madebyollin/sdxl-vae-fp16-fix"

DEVICE_CUDA = torch.device("cuda:0")
NUM_IMAGES_PER_PROMPT = 1

SDXL_NOISE_OFFSET = 0.0357

DDIM_STEPS = 16

height, width = 1024, 768

prompt = "a photo of lemonade"
negative_prompt = ""

precision = "fp16"


def flush():
    torch.cuda.empty_cache()
    gc.collect()


@torch.no_grad()
def main():
    weight_dtype = config_util.parse_precision(precision)

    (
        tokenizers,
        text_encoders,
        unet,
        noise_scheduler,
    ) = model_util.load_models_xl(
        SDXL_V09_MODEL_NAME,
        scheduler_name="ddim",
    )

    for text_encoder in text_encoders:
        text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
        text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.eval()

    add_time_ids = train_util.get_add_time_ids(
        height,
        width,
        dynamic_crops=False,
        dtype=weight_dtype,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    positive_embeds, positive_pooled_embeds = train_util.encode_prompts_xl(
        tokenizers,
        text_encoders,
        [prompt],
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
    )
    negative_embeds, negative_pooled_embeds = train_util.encode_prompts_xl(
        tokenizers,
        text_encoders,
        [negative_prompt],
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
    )

    text_embeds = train_util.concat_embeddings(
        negative_embeds,
        positive_embeds,
        1,
    )
    add_text_embeds = train_util.concat_embeddings(
        negative_pooled_embeds,
        positive_pooled_embeds,
        1,
    )
    add_time_ids = train_util.concat_embeddings(
        add_time_ids,
        add_time_ids,
        1,
    )

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        del tokenizer, text_encoder

    flush()

    noise_scheduler.set_timesteps(DDIM_STEPS, device=DEVICE_CUDA)

    latents = train_util.get_initial_latents(noise_scheduler, 1, height, width, 1)
    latents = latents * noise_scheduler.init_noise_sigma
    latents = train_util.apply_noise_offset(latents, SDXL_NOISE_OFFSET)
    latents = latents.to(DEVICE_CUDA, dtype=weight_dtype)

    # ちょっとデノイズされれたものが返る
    latents = train_util.diffusion_xl(
        unet,
        noise_scheduler,
        latents,  # 単純なノイズのlatentsを渡す
        text_embeddings=text_embeds,
        add_text_embeddings=add_text_embeds,
        add_time_ids=add_time_ids,
        total_timesteps=DDIM_STEPS,
        start_timesteps=0,
        guidance_scale=7,
    )

    del (
        unet,
        noise_scheduler,
        positive_embeds,
        positive_pooled_embeds,
        negative_embeds,
        negative_pooled_embeds,
        add_time_ids,
    )

    vae = AutoencoderKL.from_pretrained(SDXL_VAE_FP16_FIX_MODEL_NAME).to(
        DEVICE_CUDA, dtype=weight_dtype
    )
    vae.eval()

    vae.post_quant_conv.to(latents.dtype)
    vae.decoder.conv_in.to(latents.dtype)
    vae.decoder.mid_block.to(latents.dtype)

    image_tensor = vae.decode(
        latents / vae.config.scaling_factor,
    ).sample
    images = (image_tensor / 2 + 0.5).clamp(0, 1)  # denormalize らしい

    for i, image in enumerate(images):
        torchvision.utils.save_image(
            image,
            f"output_{i}.png",
        )

    flush()

    print("Done.")


if __name__ == "__main__":
    main()
