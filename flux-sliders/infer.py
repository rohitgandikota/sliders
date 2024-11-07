import os
import random
import torch
from pathlib import Path
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig
from utils.lora import LoRANetwork
from utils.custom_flux_pipeline import FluxPipeline


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, 
    subfolder: str = "text_encoder",
    device: str = "cuda"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder=subfolder,
        device_map=device
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(pretrained_model_name_or_path, class_one, class_two, weight_dtype, device="cuda"):
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        torch_dtype=weight_dtype,
        device_map=device
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder_2", 
        torch_dtype=weight_dtype,
        device_map=device
    )
    return text_encoder_one, text_encoder_two


def run_inference(
    pretrained_model_path,
    lora_path,
    target_prompt,
    output_dir,
    device="cuda",
    weight_dtype=torch.bfloat16,
    height=512,
    width=512,
    guidance_scale=3.5,
    num_inference_steps=30,
    max_sequence_length=512,
    num_images=1
):
    # Load base models
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer", torch_dtype=weight_dtype
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        pretrained_model_path, subfolder="tokenizer_2", torch_dtype=weight_dtype
    )

    # Load text encoders
    text_encoder_cls_one = import_model_class_from_model_name_or_path(pretrained_model_path, device=device)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        pretrained_model_path, subfolder="text_encoder_2", device=device
    )
    text_encoder_one, text_encoder_two = load_text_encoders(
        pretrained_model_path, text_encoder_cls_one, text_encoder_cls_two, weight_dtype, device
    )

    # Load other models
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path, subfolder="vae", torch_dtype=weight_dtype
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler", torch_dtype=weight_dtype
    )

    # Move models to device
    vae.to(device)
    transformer.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)

    # Create pipeline
    pipe = FluxPipeline(
        noise_scheduler,
        vae,
        text_encoder_one,
        tokenizer_one,
        text_encoder_two,
        tokenizer_two,
        transformer,
    )
    pipe.set_progress_bar_config(disable=True)

    # Load LoRA networks
    networks = {}
    
    lora_files = sorted(Path(lora_path).glob("slider_*.pt"))
    print(f'Loading {len(lora_files)} sliders from {lora_path}')
    for i, lora_file in enumerate(lora_files):
        networks[i] = LoRANetwork(
            transformer,
            rank=16,  # These values should match your training configuration
            multiplier=1.0,
            alpha=1,
            train_method='xattn'
        )
        networks[i].load_state_dict(torch.load(str(lora_file)))
        networks[i].to(device, dtype=weight_dtype)

    # Generate images
    os.makedirs(output_dir, exist_ok=True)
    seeds = [random.randint(0, 2**15) for _ in range(num_images)]
    print(f'Generating {num_images} images with {len(networks)} sliders')
    for net in networks:
        print(f'Generating with Slider {net}')
        for idx in range(num_images):
            seed = seeds[idx]
            for slider_scale in [-5, -1, 0, 1, 5]:
                networks[net].set_lora_slider(scale=slider_scale)
                with torch.no_grad():
                    image = pipe(
                        target_prompt,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        max_sequence_length=max_sequence_length,
                        num_images_per_prompt=1,
                        generator=torch.Generator().manual_seed(seed),
                        from_timestep=0,
                        till_timestep=None,
                        output_type='pil',
                        network=networks[net],
                        skip_slider_timestep_till=0,
                    )
                img_filename = f"slider_{net}_seed_{seed}_scale_{slider_scale}.png"
                image.images[0].save(os.path.join(output_dir, img_filename))
    
    print("Inference completed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--target_prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=1)
    args = parser.parse_args()

    run_inference(
        pretrained_model_path=args.pretrained_model_path,
        lora_path=args.lora_path,
        target_prompt=args.target_prompt,
        output_dir=args.output_dir,
        num_images=args.num_images
    )