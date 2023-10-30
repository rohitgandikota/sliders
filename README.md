# sliders
Project repository for LoRA sliders for diffusion models


## Training a LoRa
To train an age slider - go to `data/prompts.yaml` and edit the `target=person` and `positive=old person` and `unconditional=young person` (opposite of positive) and `neutral=person` and `action=enhance` with `guidance=4`.
If you do not want your edit to be targetted to person replace it with any target you want (eg. dog) or if you need it global replace `person` with `""`  <br>
Finally, run the command:
```
python train-lora.py --attributes 'male, female' --name 'ageslider' --rank 4 --alpha 1 --config_file 'data/config.yaml'
```

`--attributes` argument is used to disentangle concepts from the slider. For instance age slider makes all old people male (so instead add the `"female, male"` attributes to allow disentanglement)
