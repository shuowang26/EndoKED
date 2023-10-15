import pandas as pd
from glob import glob
import numpy as np
from PIL import Image
import torch
import os
from torchvision import transforms as T
# from . import transforms
import transforms
import imageio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import csv


def load_img_label_info_from_csv(csv_dir,pos_num=2,):

    img_path_list_pos = []
    label = []
    csv_path_list = glob(os.path.join(csv_dir,'*'))
    for item_path in csv_path_list:
        # read csv_item
        reader = pd.read_csv(item_path)
        sorted_reader = reader.sort_values(by='1')
        if 'gt1' in item_path:
            for num in range(pos_num):
                idx_ = -1 - num
                idx = sorted_reader.index[idx_]
                path_name = sorted_reader['0'][idx].replace('/home/ubuntu/Data/database/中山/','/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/中山/')
                img_path_list_pos.append(path_name)
                # label.append(one_hot_pos)
                label.append(np.array([1.0]))

    
    # img_path_list_pos = np.array(img_path_list_pos)
    with open ('/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/polyp_zhongshan_pos/selected_polyp_path.txt','a') as f:
        for item in img_path_list_pos:
            f.write(f'{item}\n')
    return 


if __name__ == "__main__":

    root_dir_train = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/network_MIL_prediction_results/train/'
    root_dir_test = "/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/CVC-ClinicDB_datasets/CVC-ClinicDB_datasets/CVC-ClinicDB_PNG_datasets/"
    
    positive_polyp_save_dir = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/中山_polyp_selected/'
    img_path_list_train, label_train = load_img_label_info_from_csv(root_dir_train)