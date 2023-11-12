# sliders
Project repository for LoRA sliders for diffusion models

## Text-based Sliders
### Training SD-1.x and SD-2.x LoRa
To train an age slider - go to `train-scripts/textsliders/data/prompts.yaml` and edit the `target=person` and `positive=old person` and `unconditional=young person` (opposite of positive) and `neutral=person` and `action=enhance` with `guidance=4`. <br>
If you do not want your edit to be targetted to person replace it with any target you want (eg. dog) or if you need it global replace `person` with `""`  <br>
Finally, run the command:
```
python train-lora.py --attributes 'male, female' --name 'ageslider' --rank 4 --alpha 1 --config_file 'data/config.yaml'
```

`--attributes` argument is used to disentangle concepts from the slider. For instance age slider makes all old people male (so instead add the `"female, male"` attributes to allow disentanglement)


#### Evaluate 
To evaluate your trained models use the notebook `sliders-inference-clean.ipynb`


### Training SD-XL
To train sliders for SD-XL, use the script `train-lora-xl.py`. The setup is same as SDv1.4

```
python train-lora-xl.py --attributes 'male, female' --name 'agesliderXL' --rank 4 --alpha 1 --config_file 'data/config-xl.yaml'
```

#### Evaluate 
To evaluate your trained models use the notebook `XL-sliders-inference.ipynb`


## Image-based Sliders
### Training SD-1.x and SD-2.x LoRa
To train image based sliders, you need to create a ~4-6 pairs of image dataset (before/after edit for desired concept). Save the before images and after images separately. You can also create a dataset with varied intensity effect and save them differently. 

To train an image slider for eye size - go to `train-scripts/imagesliders/data/config.yaml` and edit the `target=eye` and `positive='eye'` and `unconditional=''` and `neutral=eye` and `action=enhance` with `guidance=4`. <br>
If you want the diffusion model to figure out the edit concept - leave `target, postive, unconditional, neutral` as `''`<br>
Finally, run the command:
```
python train-lora-scale.py --name 'eyeslider' --rank 4 --alpha 1 --config_file 'data/config.yaml' --folder_main 'datasets/eyesize/' --folders 'bigsize, smallsize' --scales '1, -1' 
```

### Training SD-XL
To train image sliders for SD-XL, use the script `train-lora-scale-xl.py`. The setup is same as SDv1.4

```
python train-lora-scale-xl.py --name 'eyesliderXL' --rank 4 --alpha 1 --config_file 'data/config-xl.yaml' --folder_main 'datasets/eyesize/' --folders 'bigsize, smallsize' --scales '1, -1' 
```
