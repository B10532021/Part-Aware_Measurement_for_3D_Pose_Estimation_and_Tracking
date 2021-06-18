# Part-aware Measurement for Robust Multi-View Multi-Human 3D Pose Estimation and Tracking

## Installation

 - Python 3.6+

 - Cuda 9.0

 - Cudnn 7

 - gcc 5 & g++ 5 (for Ubuntu 18.04)
```
$ sudo apt install gcc-5 g++-5
$ sudo ln -s /usr/bin/gcc-6 /usr/local/bin/gcc
$ sudo ln -s /usr/bin/g++-6 /usr/local/bin/g++
```

 - Conda Env
```
$ conda create -n venv python=3.6
$ conda activate venv
$ conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
$ pip install tensorflow_gpu==1.9.0
$ pip install -r requirements.txt
```
 
 - Git
```
$ sudo apt install git
```

# Data preparation
Download datasets:
1. Campus (http://campar.in.tum.de/Chair/MultiHumanPose)
2. Shelf (http://campar.in.tum.de/Chair/MultiHumanPose)
3. CMU Panoptic (https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox)

The directory tree should look like below:
```
${ROOT}
    |-- CatchImage
        |-- CampusSeq1
        |   |-- Camera0
        |   |-- Camera1
        |   |-- Camera2
        |   |-- camera_parameter.pickle
        |   |-- actorsGT.mat
        |-- Shelf
        |   |-- Camera0
        |   |-- ...
        |   |-- Camera4
        |   |-- camera_parameter.pickle
        |   |-- actorsGT.mat
        |-- Panoptic
        |   |-- 160906_pizza1
            |   |-- 00_03 # hdImgs folder of 03 camera
            |   |-- 00_06 # hdImgs folder of 06 camera
            |   |-- ...
            |   |-- camera_parameter.pickle
            |   |-- hdPose_stage1_coco19
            |-- ...
```

## Backend Models
Backend models, which is not our works, are released codes from others. We only did some small modifications to fit the format of our input/output.
Put models in {ROOT}/src/leb/backend
1. YOLOv3
    - origin: https://github.com/ZQPei/deep_sort_pytorch
    - [modification](https://drive.google.com/drive/folders/16cPluYbBahb1wzN5CKfeBuJw9mzrFUJT?usp=sharing) 
2. HRNet
    - origin: https://github.com/stefanopini/simple-HRNet
    - [modification](https://drive.google.com/drive/folders/19lLnky0JsV6IyfB4x-LTz2Azx4cuEi-L?usp=sharing) 

## Run Codes
### Demo
```bash
$cd src
python -W ignore testmodel.py --dataset CampusSeq1 # For Campus
python -W ignore testmodel.py --dataset Shelf # For Shelf
python -W ignore testmodel.py --dataset Panoptic # For Panoptic (sub-dataset can be modified in config)
```
### Evaluation
```bash
$cd src
python -W ignore evalmodel.py --dataset CampusSeq1 
python -W ignore evalmodel.py --dataset Shelf 
python -W ignore evalmodel.py --dataset Panoptic 
```