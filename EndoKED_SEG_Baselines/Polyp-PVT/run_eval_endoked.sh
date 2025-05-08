#!/bin/bash

source activate endoked

gpu=0
# test_path=/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/
test_path=./data/polyp_public/TestDataset

file=./EndoKED_SEG_Baselines/Polyp-PVT-main/test_ENDOKED.py

pt_path=./EndoKED_SEG_Baselines/Polyp-PVT-main/model_pth/53PolypPVT.pth
CUDA_VISIBLE_DEVICES=$gpu python $file --pt_path=$pt_path --testdata_path=$test_path

# pt_path=./EndoKED_SEG_Baselines/Polyp-PVT-main/logs/Train_on_ZhongshanandKvasirandDB/2024-04-09-14-55-11/26PolypPVT-best.pth
# CUDA_VISIBLE_DEVICES=$gpu python $file --pt_path=$pt_path --testdata_path=$test_path