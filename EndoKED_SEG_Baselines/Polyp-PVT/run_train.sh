#!/bin/bash

source activate endoked

gpu=$1
nproc_per_node=1
master_port=29740

tag=loaded_from_endoked
load_pretrained=True
cpt_path=./EndoKED_SEG_Baselines/Polyp-PVT-main/logs/Train_on_Zhongshan/2024-04-20-13-44-19_train_on_zhongshan/24PolypPVT-best.pth

file=./EndoKED_SEG_Baselines/Polyp-PVT-main/Train.py
root=./data/polyp_public/TrainDataset/ #polygen_train/
val_root=./data/polyp_public/TestDataset/ #polygen_test/

dataset=Train_on_KvasirandDB
CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node \
--master_port=$master_port $file --train_path=$root --test_path=$val_root --dataset=$dataset \
--tag=$tag --load_pretrained=$load_pretrained --cpt_path=$cpt_path