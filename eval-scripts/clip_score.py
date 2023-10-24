from PIL import Image
import requests
import os, glob
import pandas as pd
import numpy as np
import re
import argparse
from transformers import CLIPProcessor, CLIPModel


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'clipScore',
                    description = 'Generate CLIP score for images')
    parser.add_argument('--im_path', help='path for images', type=str, required=True)
    parser.add_argument('--prompt', help='prompt to check clip score against', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--till_case', help='continue generating from case_number', type=int, required=False, default=1000000)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    
    args = parser.parse_args()

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def sorted_nicely( l ):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
        return sorted(l, key = alphanum_key)


    path = args.im_path #'/share/u/rohit/www/final_erase/coco/'
    model_names = os.listdir(path)
    model_names = [m for m in model_names if 'all' not in m and '.csv' not in m]
    csv_path = args.prompts_path #'/share/u/rohit/erase-closed/prompts_dir/erased'
    save_path = ''
    prompt = args. prompt.strip()
    print(f'Eval agaisnt prompt: {prompt}')
    model_names.sort()
    print(model_names)
    df = pd.read_csv(csv_path)
    for model_name in model_names:
        print(model_name)
#         csv_path = f'/share/u/rohit/erase-closed/data/coco_30k.csv'
        im_folder = os.path.join(path, model_name)
#         df = pd.read_csv(csv_path)
        images = os.listdir(im_folder)
        images = sorted_nicely(images)
        ratios = {}
        model_name = model_name.replace('half','0.5')
        df[f'clip_{model_name}'] = np.nan
        for image in images:
            try:
                case_number = int(image.split('_')[0].replace('.png',''))
                if case_number not in list(df['case_number']):
                    continue
                im = Image.open(os.path.join(im_folder, image))
                inputs = processor(text=[prompt], images=im, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                clip_score = outputs.logits_per_image[0][0].detach().cpu() # this is the image-text similarity score
                ratios[case_number] = ratios.get(case_number, []) + [clip_score]
#                 print(image, clip_score)
            except:
                pass
        for key in ratios.keys():
            df.loc[key,f'clip_{model_name}'] = np.mean(ratios[key])
#         df = df.dropna(axis=0)
        print(f"Mean CLIP score: {df[f'clip_{model_name}'].mean()}")
        print('-------------------------------------------------')
        print('\n')
    df.to_csv(f'{path}/clip_scores.csv', index=False)