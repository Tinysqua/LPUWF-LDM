# **LPUWF-LDM**

## Overall structure of the LPUWF-LDM:
![model](img/Model.png)

This code is the pytorch implementation of LPUWF-LDM

## Pre-requisties
- Linux
- numpy==1.26.3
- torch==2.0.0+cu118
- tqdm==4.66.2
- accelerate==0.27.2
- diffusers==0.18.2
- monai==1.3.0
- monai-generative==0.2.3
- lpips==0.1.4
- tensorboard==2.15.1

## How to train
### Example Data
We take 6 pairs images to be an example. They put in directory: visible_test. For a pair, x.bmp means UWF-SLO, x.1.bmp means early-phase UWF-FA and x.2.bmp means late-phase
### First stage: Conditional VAE (C-VAE)
```
cd LPUWF-LDM
accelerate launch --multi_gpu --mixed_precision=fp16 --main_process_port 29500 \
ldm/1024_vae_addition.py
```
To configure the options of this stage, reference to ./config/1024_vae_addition_test_config.yaml

### Second stage: Diffusion 
```
cd LPUWF-LDM
accelerate launch --multi_gpu --mixed_precision=fp16 --main_process_port 29500 \
sd/control_net_with_sobel_small.py
```
To configure the options of this stage, reference to ./config/control_net_config.yaml
