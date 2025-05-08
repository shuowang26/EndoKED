#!/bin/bash

source activate endoked

gpu=0
exp_tag=eval_all

# test_path=/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/
test_path=./data/polyp_public/TestDataset
file=./EndoKED_SEG_Baselines/Polyp-PVT-main/test_polyp_polygen.py

pt_path=./EndoKED_SEG_Baselines/Polyp-PVT-main/logs/Train_on_KvasirandDB/2024-04-21-10-56-31_loaded_from_endoked/93PolypPVT-best.pth
CUDA_VISIBLE_DEVICES=$gpu python $file --pt_path=$pt_path --testdata_path=$test_path --exp_tag=$exp_tag
