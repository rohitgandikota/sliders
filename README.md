# Concept Sliders
###  [Project Website](https://sliders.baulab.info) | [Arxiv Preprint](https://arxiv.org/pdf/2311.12092.pdf) | [Trained Sliders](https://sliders.baulab.info/weights/xl_sliders/)<br>
Official code implementation of "Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models"

<div align='center'>
<img src = 'images/main_figure.png'>
</div>
## Colab Demo
Try out our colab demo here 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohitgandikota/sliders/blob/main/XL-sliders-inference.ipynb)

## Setup
To set up your python environment:
```
conda create -n sliders python=3.9
conda activate sliders

git  clone https://github.com/rohitgandikota/sliders.git
cd sliders
pip install -r requirements.txt
```
## Textual Concept Sliders
### Training SD-1.x and SD-2.x LoRa
To train an age slider - go to `train-scripts/textsliders/data/prompts.yaml` and edit the `target=person` and `positive=old person` and `unconditional=young person` (opposite of positive) and `neutral=person` and `action=enhance` with `guidance=4`. <br>
If you do not want your edit to be targetted to person replace it with any target you want (eg. dog) or if you need it global replace `person` with `""`  <br>
Finally, run the command:
```
python trainscripts/textsliders/train_lora.py --attributes 'male, female' --name 'ageslider' --rank 4 --alpha 1 --config_file 'trainscripts/textsliders/data/config.yaml'
```

`--attributes` argument is used to disentangle concepts from the slider. For instance age slider makes all old people male (so instead add the `"female, male"` attributes to allow disentanglement)


#### Evaluate 
To evaluate your trained models use the notebook `SD1-sliders-inference.ipynb`


### Training SD-XL
To train sliders for SD-XL, use the script `train-lora-xl.py`. The setup is same as SDv1.4

```
python trainscripts/textsliders/train-lora-xl.py --attributes 'male, female' --name 'agesliderXL' --rank 4 --alpha 1 --config_file 'data/config-xl.yaml'
```

#### Evaluate 
To evaluate your trained models use the notebook `XL-sliders-inference.ipynb`


## Visual Concept Sliders
### Training SD-1.x and SD-2.x LoRa
To train image based sliders, you need to create a ~4-6 pairs of image dataset (before/after edit for desired concept). Save the before images and after images separately. You can also create a dataset with varied intensity effect and save them differently. 

To train an image slider for eye size - go to `train-scripts/imagesliders/data/config.yaml` and edit the `target=eye` and `positive='eye'` and `unconditional=''` and `neutral=eye` and `action=enhance` with `guidance=4`. <br>
If you want the diffusion model to figure out the edit concept - leave `target, postive, unconditional, neutral` as `''`<br>
Finally, run the command:
```
python trainscripts/imagesliders/train_lora-scale.py --name 'eyeslider' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config.yaml' --folder_main 'datasets/eyesize/' --folders 'bigsize, smallsize' --scales '1, -1' 
```
For this to work - you need to store your before images in `smallsize` and after images in `bigsize`. The corresponding paired files in both the folders should have same names. Both these subfolders should be under `datasets/eyesize`. Feel free to make your own datasets in your own named conventions.
### Training SD-XL
To train image sliders for SD-XL, use the script `train-lora-scale-xl.py`. The setup is same as SDv1.4

```
python trainscripts/imagesliders/train_lora-scale-xl.py --name 'eyesliderXL' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config-xl.yaml' --folder_main '/share/u/rohit/imageXLdataset/eyesize_data/'
```

## Citing our work
The preprint can be cited as follows
```
@article{gandikota2023sliders,
  title={Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Tingrui Zhou and Antonio Torralba and David Bau},
  journal={arXiv preprint arXiv:2311.12092},
  year={2023}
}
```
