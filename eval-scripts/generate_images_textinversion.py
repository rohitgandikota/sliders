from diffusers import StableDiffusionPipeline
import pandas as pd
import torch
import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Generate Text Inversion Images',)
    
    parser.add_argument('--model_name', help='path to custom model', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv prompts', type=str, required=True)
    parser.add_argument('--token', help='path to csv prompts', type=str, required=True)
    args = parser.parse_args()
    model_id = args.model_name #"/share/u/rohit/textual_inversion_eyebrows-v1-4"
    custom_token = args.token #'<sks-eyebrows>'


    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")

    df = pd.read_csv(args.prompts_path)

    prompts = list(df.prompt)
    seeds = list(df.evaluation_seed)
    case_numbers = list(df.case_number)
    file = os.path.basename(model_id)

    os.makedirs(f'/share/u/rohit/www/text_inversion/{file}/',exist_ok=True)
    for idx,prompt in enumerate(prompts):

        prompt += f" with {custom_token}"
        case_number = case_numbers[idx]
        generator = torch.manual_seed(seeds[idx])
        images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=5).images
        for i, im in enumerate(images):
            im.save(f'/share/u/rohit/www/text_inversion/{file}/{case_number}_{i}.png')