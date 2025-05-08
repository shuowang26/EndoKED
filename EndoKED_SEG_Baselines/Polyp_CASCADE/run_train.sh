#!/bin/bash

source activate endoked

gpu=$1
nproc_per_node=1
master_port=$2

load_pretrained=True
file=./EndoKED_SEG_Baselines/CASCADE-main/train_polyp.py
root=./data/polyp_public/TrainDataset/ #polygen_train/
val_root=./data/polyp_public/TestDataset/ #polygen_test/

dataset=Train_on_KvasirandDB #Train_on_Zhongshan #Train_on_KvasirandDB

seed=0
cpt_path=./EndoKED_SEG_Baselines/CASCADE-main/logs/Train_on_Zhongshan/2024-05-03-14-43-08_loaded_from_endoked_seed0_lr6e-4/PolypPVT-CASCADE51_best.pth
tag=loaded_from_endoked_1
CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node \
--master_port=$master_port $file --train_path=$root --test_path=$val_root --dataset=$dataset \
--tag=$tag --load_pretrained=$load_pretrained --cpt_path=$cpt_path --seed=$seed

cpt_path=./EndoKED_SEG_Baselines/CASCADE-main/logs/Train_on_Zhongshan/2024-05-03-14-44-13_loaded_from_endoked_seed0_lr4e-4/PolypPVT-CASCADE8_best.pth
tag=loaded_from_endoked_2
CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node \
--master_port=$master_port $file --train_path=$root --test_path=$val_root --dataset=$dataset \
--tag=$tag --load_pretrained=$load_pretrained --cpt_path=$cpt_path --seed=$seed

cpt_path=./EndoKED_SEG_Baselines/CASCADE-main/logs/Train_on_Zhongshan/2024-05-03-14-45-21_loaded_from_endoked_seed0_lr6e-5/PolypPVT-CASCADE2_best.pth
tag=loaded_from_endoked_3
CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node \
--master_port=$master_port $file --train_path=$root --test_path=$val_root --dataset=$dataset \
--tag=$tag --load_pretrained=$load_pretrained --cpt_path=$cpt_path --seed=$seed