#  Lesion-Aware Dynamic Kernel for Polyp Segmentation

##  Introduction

This repository contains the PyTorch implementation of:

Lesion-Aware Dynamic Kernel for Polyp Segmentation, MICCAI 2022.

##  Requirements

* torch
* torchvision 
* tqdm
* opencv
* scipy
* skimage
* PIL
* numpy

##  Usage

####  1. Training

```bash
python train.py  --root /path-to-project  --mode train
--train_data_dir /path-to-train_data   --valid_data_dir  /path-to-valid_data
```



####  2. Inference

```bash
python test.py  --root /path-to-project  --mode test  --load_ckpt checkpoint  
--test_data_dir  /path-to-test_data
```



##  Citation

If you feel this work is helpful, please cite our paper

```
@inproceedings{zhang2022lesion,
  title={Lesion-Aware Dynamic Kernel for Polyp Segmentation},
  author={Zhang, Ruifei and Lai, Peiwen and Wan, Xiang and Fan, De-Jun and Gao, Feng and Wu, Xiao-Jian and Li, Guanbin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={99--109},
  year={2022},
  organization={Springer}
}
```





