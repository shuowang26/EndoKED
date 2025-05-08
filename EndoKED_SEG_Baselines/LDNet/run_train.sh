#!/bin/bash

source activate endoked

gpu=$1
nproc_per_node=1
master_port=29749

tag=loaded_from_endoked
load_pretrained=False
cpt_path=./EndoKED_SEG_Baselines/LDNet-main/logs/Train_on_Zhongshan/2024-04-20-13-50-55/Train_on_Zhongshan_13_Best.pt

file=./EndoKED_SEG_Baselines/LDNet-main/train.py
root=./data/polyp_public/TrainDataset/ #polygen_train/
val_root=./data/polyp_public/TestDataset/ #polygen_test/

dataset=Train_on_KvasirandDB
seed=10
# dataset=Train_on_Zhongshan
CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node \
--master_port=$master_port $file --train_data_dir=$root --val_root=$val_root --dataset=$dataset \
--tag=$tag --load_pretrained=$load_pretrained --cpt_path=$cpt_path --seed=$seed

seed=100
# dataset=Train_on_Zhongshan
CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node \
--master_port=$master_port $file --train_data_dir=$root --val_root=$val_root --dataset=$dataset \
--tag=$tag --load_pretrained=$load_pretrained --cpt_path=$cpt_path --seed=$seed