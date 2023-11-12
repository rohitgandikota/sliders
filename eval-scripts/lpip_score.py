from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy
import os
import pandas as pd
import argparse
import lpips


# desired size of the output image
imsize = 64
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    image = (image-0.5)*2
    return image.to(torch.float)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'LPIPS',
                    description = 'Takes the path to two images and gives LPIPS')
    parser.add_argument('--im_path', help='path to original image', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv prompts', type=str, required=True)
    parser.add_argument('--true', help='path to true SD images', type=str, required=True)
    
    loss_fn_alex = lpips.LPIPS(net='alex')
    args = parser.parse_args()
    
    true = args.true
    models = os.listdir(args.im_path)
    models = [m for m in models if m not in [true,'all'] and '.csv' not in m]
    
    original_path = os.path.join(args.im_path,true)
    df_prompts = pd.read_csv(args.prompts_path)
    for model_name in models:
        edited_path = os.path.join(args.im_path,model_name)
        file_names = [name for name in os.listdir(edited_path) if '.png' in name]
        model_name = model_name.replace('half','0.5')
        df_prompts[f'lpips_{model_name}'] = df_prompts['case_number'] *0
        for index, row in df_prompts.iterrows():
            case_number = row.case_number
            files = [file for file in file_names if file.startswith(f'{case_number}_')]
            lpips_scores = []
            for file in files:
                print(file)
                try:
                    original = image_loader(os.path.join(original_path,file))
                    edited = image_loader(os.path.join(edited_path,file))

                    l = loss_fn_alex(original, edited)
                    
                    lpips_scores.append(l.item())
                except Exception:
                    print('No File')
                    pass
            print(f'Case {case_number}: {np.mean(lpips_scores)}')
            df_prompts.loc[index,f'lpips_{model_name}'] = np.mean(lpips_scores)
    df_prompts.to_csv(os.path.join(args.im_path, f'lpips_score.csv'), index=False)
# python eval-scripts/lpips_eval.py --original_path '/share/u/rohit/www/closed_form/niche_short/original/' --csv_path '/share/u/rohit/erase-closed/data/short_niche_art_prompts.csv' --save_path '/share/u/rohit/www/closed_form/niche_short/' --edited_path '/share/u/rohit/www/closed_form/niche_short/erasing-ThomasKinkade-with-preservation/'