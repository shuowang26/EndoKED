#!/bin/bash

source activate endoked
gpu=0
exp_tag=xxxxx

# test_path=/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/
test_path=./data/polyp_public/TestDataset
file=./EndoKED_SEG_Baselines/PIDNet/test_polyp.py

pt_path=./EndoKED_SEG_Baselines/PIDNet/logs/Train_on_KvasirandDB/2024-05-19-01-56-15_Train_on_KvasirandDB/PolypPVT-CASCADE72_best.pth
CUDA_VISIBLE_DEVICES=$gpu python $file --pth_path=$pt_path --testdata_path=$test_path --exp_tag=$exp_tag

