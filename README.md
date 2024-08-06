# ELSSL:Innovative Deep Learning Approaches for High-Precision Segmentation and Characterization of Sandstone Pore Structures in Reservoirs
by Limin Suo , Zhaowei Wang , Hailong Liu , Likai Cui and Xianda Sun
## Introduction
This repository is the Pytorch implementation of "Innovative Deep Learning Approaches for High-Precision Segmentation and Characterization of Sandstone Pore Structures in Reservoirs"
## Requirements
The specific configuration is as follows:

* Windows 10
* Nvidia GeForce 3080Ti
  
Some important required packages include:

* CUDA 11.1
* Pytorch == 1.10.0
* Python == 3.7
* Some basic python packages such as Opencv-python, Numpy, Scikit-image,  Scipy ......

## Usage
1. Use StyleGAN2-ADA to generate a large number of high-resolution two-dimensional sandstone CT grayscale images
```
cd stylegan2-ada
python train.py
python gen.py
```
3. Train segmentation model
```
cd elssl
python pre_train.py
python train.py
```
3. Test the model
```
cd elssl
python test.py
```

## Acknowledgement

We extend our sincere gratitude to the NVIDIA Research team for their pioneering work on StyleGAN, StyleGAN2, and StyleGAN2-ADA. Their groundbreaking research has significantly contributed to the field of generative adversarial networks and has set a new benchmark for image synthesis.

## Note
* The repository is being updated.
* Contact: Zhaowei Wang (wangzhaowei@byau.edu.cn)
