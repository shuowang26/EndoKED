#! /bin/bash

source activate endoked

gpu=0
exp_tag=eval_all

# test_path=/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/
test_path=./data/polyp_public/TestDataset
file=./EndoKED_SEG_Baselines/LDNet-main/test_polyp_polygen.py

pt_path=./EndoKED_SEG_Baselines/LDNet-main/logs/Train_on_KvasirandDB/2024-04-21-05-33-40_loaded_from_endoked/Train_on_KvasirandDB_Best.pt
CUDA_VISIBLE_DEVICES=$gpu python $file --pt_path=$pt_path --testdata_path=$test_path --exp_tag=$exp_tag


# pt_path=./EndoKED_SEG_Baselines/LDNet-main/logs/Train_on_ZhongshanandKvasirandDB/2024-04-10-10-05-49/Train_on_ZhongshanandKvasirandDB_Best.pt
# CUDA_VISIBLE_DEVICES=$gpu python $file --pt_path=$pt_path --testdata_path=$test_path