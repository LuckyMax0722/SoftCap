# SoftCap: Dense Captioning for 3D Scenes with SparseConv

<p align="center"><img src="demo/Vis_Room.jpg" width="600px"/></p>

## Introduction
We propose a model using SoftGroup based detection backbone. With sparse convolution and soft grouping mechanism, 
better detection performance and denser object features can be achieved, which enables the later language model to 
generate more reliable captions. A message passing graph model and an attention mechanism are used to aggregate object 
features with relational information. Our method can effectively localize and describe objects in 3D scenes and 
outperforms the existing baseline method with a significant improvement.

## Installation
1. Environment requirements

* Python 3.8
* Pytorch 1.11
* CUDA 11.3

The following installation guild suppose ``python=3.8`` ``pytorch=1.11`` and ``cuda=11.3``. You may change them according to your system.

Create a conda virtual environment and activate it.
```
conda create -n softgroup python=3.8
conda activate softgroup
```

2. Clone the repository.
```
git clone https://github.com/LuckyMax0722/SoftCap.git
```


3. Install the dependencies.
```
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install spconv-cu113
pip install -r requirements.txt
```

4. Install build requirement.

```
sudo apt-get install libsparsehash-dev
```


For detail information, please refer to [SoftGroup/Installation](https://github.com/thangvubk/SoftGroup/blob/main/docs/installation.md)

## Data Preparation
Please refer to [SoftGroup/Data Preparation/ScanNet](https://github.com/thangvubk/SoftGroup/blob/main/dataset/README.md#scannet-v2-dataset)
and [Scan2Cap/Data Preparation](https://github.com/daveredrum/Scan2Cap/blob/main/README.md#data-preparation)

## Pretrained Models
For SoftGroup pretrained models, please refer to [SoftGroup/Pretrained Models](https://github.com/thangvubk/SoftGroup#instance-segmentation)

For SoftGroup pretrained models, please refer to following table

|  Model  | Network Architecture | Loss  |                                           Download                                         |
|:-------:|:--------------------:|:-----:|:-------------------------------------------------------------------------------------------:|
| SoftCap | SoftGroup + RG + CAC |  CE   | [model](https://drive.google.com/file/d/1-f7I6-eIma4OilBON928N6mVcYbhiUFP/view?usp=sharing) |
| SoftCap | SoftGroup + RG + CAC | CIDEr | [model](https://drive.google.com/file/d/1-f7I6-eIma4OilBON928N6mVcYbhiUFP/view?usp=sharing) |

## Training
```shell
python scripts/train.py
```

## Visualization
```shell
python scripts/eval.py
```

