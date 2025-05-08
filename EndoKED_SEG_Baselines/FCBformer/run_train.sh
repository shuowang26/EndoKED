#!/bin/bash

source activate endoked

gpu=$1
nproc_per_node=1
master_port=29767

tag=loaded_from_endoked
load_pretrained=True
cpt_path=./EndoKED_SEG_Baselines/FCBFormer/logs/Train_on_Zhongshan/2024-04-21-10-55-12_train_on_zhongshan/Train_on_Zhongshan17_Best.pt

file=./EndoKED_SEG_Baselines/FCBFormer/train_endo.py
root=./data/polyp_public/TrainDataset/
val_root=./data/polyp_public/TestDataset/

dataset=Train_on_KvasirandDB

CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node \
--master_port=$master_port $file --train_path=$root --test_path=$val_root --dataset=$dataset \
--tag=$tag --load_pretrained=$load_pretrained --cpt_path=$cpt_path