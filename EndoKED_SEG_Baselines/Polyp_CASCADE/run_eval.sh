#!/bin/bash

source activate endoked
gpu=0
exp_tag=eval_all


# test_path=/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/
test_path=./data/polyp_public/TestDataset
file=./EndoKED_SEG_Baselines/CASCADE-main/test_polyp_polygen.py

pt_path=./EndoKED_SEG_Baselines/CASCADE-main/logs/Train_on_KvasirandDB/2024-05-07-01-36-35_loaded_from_endoked_7/PolypPVT-CASCADE97_best.pth
CUDA_VISIBLE_DEVICES=$gpu python $file --pth_path=$pt_path --testdata_path=$test_path --exp_tag=$exp_tag


