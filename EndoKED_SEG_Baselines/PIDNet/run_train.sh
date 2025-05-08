#!/bin/bash

source activate endoked

gpu=$1
nproc_per_node=1
master_port=$2

imgnet_pretrained=False
load_pretrained=True
cpt_path=./EndoKED_SEG_Baselines/PIDNet/logs/Train_on_Zhongshan/2024-05-18-14-45-27_train_on_Zhongshan/PolypPVT-CASCADE12_best.pth

file=/home/yzw_21110860024/yzw_workspace/code_base/endokd_rebuttal/PIDNet/train_polyp.py
root=./data/polyp_public/TrainDataset/ #polygen_train/
val_root=./data/polyp_public/TestDataset/ #polygen_test/

dataset=Train_on_KvasirandDB #Train_on_Zhongshan #Train_on_KvasirandDB

seed=0
tag=Train_on_KvasirandDB
CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node \
--master_port=$master_port $file --train_path=$root --test_path=$val_root --dataset=$dataset \
--tag=$tag --imgnet_pretrained=$imgnet_pretrained --seed=$seed --load_pretrained=$load_pretrained --cpt_path=$cpt_path