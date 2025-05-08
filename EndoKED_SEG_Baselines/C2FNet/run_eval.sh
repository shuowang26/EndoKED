#!/bin/bash
source activate endoked

gpu=0
exp_tag=eval_allxxx

test_path=./data/polyp_public/TestDataset
file=./EndoKED_SEG_Baselines/C2FNet-master/test_polyp_polygen.py

pt_path=./EndoKED_SEG_Baselines/C2FNet-master/logs/Train_on_KvasirandDB/2024-04-21-04-29-16_loaded_from_endoked/Train_on_KvasirandDB_15_Best.pt
CUDA_VISIBLE_DEVICES=$gpu python $file --pt_path=$pt_path --testdata_path=$test_path --exp_tag=$exp_tag
