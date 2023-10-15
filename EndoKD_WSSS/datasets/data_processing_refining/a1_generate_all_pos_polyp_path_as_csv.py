
###### zhongshan 阳性样本加上阈值； 加上两万例阳性样本； 加上所有阴性样本中山
import pandas as pd
from glob import glob
import numpy as np
import os
import csv
import json


def load_img_path_from_2w(public_2w_path):
    img_lst_ = glob(os.path.join(public_2w_path,'images/*'))
    img_pos_lst = []
    label_pos_lst = []
    img_neg_lst = []
    label_neg_lst = []
    label_json_lst = glob(os.path.join(public_2w_path,'labels/*'))
    for img_path, f in zip(img_lst_,label_json_lst):
        name = img_path.split('images/')[-1][:-4]
        json_path = os.path.join(public_2w_path,f'labels/{name}.json')
        if json_path in label_json_lst:
            fl = open(json_path,'r')
            data = json.load(fl)
            data_info = data['shapes'][0]
            label = data_info['label']
            if label == '0':
                label_pos_lst.append(np.array([1.0]))
                img_pos_lst.append(img_path)
            else:
                label_neg_lst.append(np.array([0.0]))
                img_neg_lst.append(img_path)


    return img_pos_lst, label_pos_lst, img_neg_lst, label_neg_lst


def load_img_and_save_pos_path(csv_dir,public_2w_path,csv_save_path, threshold=0.8,pos_num=5):

    if not os.path.exists(csv_save_path):
        zhongshan_pos_img_path_list = []
        label_name_lst = []
        # public_img_path_list = glob(f'{public_2w_path}/*')
        pub_img_pos_lst, pub_label_pos_lst, pub_img_neg_lst, pub_label_neg_lst = load_img_path_from_2w(public_2w_path)
        csv_path_list = glob(os.path.join(csv_dir,'*'))
        for item_path in csv_path_list:
            # read csv_item
            reader = pd.read_csv(item_path)
            sorted_reader = reader.sort_values(by='1')
            if 'gt1' in item_path:
                for num in range(len(sorted_reader.index)):
                    idx_ = -1 - num
                    idx = sorted_reader.index[idx_]
                    score = sorted_reader['1'][idx]
                    if score >= threshold:
                        path_name = sorted_reader['0'][idx].replace('/home/ubuntu/Data/database/中山/','/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/中山/')
                        zhongshan_pos_img_path_list.append(path_name)
                
        img_path_list = zhongshan_pos_img_path_list + pub_img_pos_lst

        with open (csv_save_path,'w') as f:
            header = ['img_path','label_name']
            writer = csv.writer(f)
            writer.writerow(header)
            for idx, item in enumerate(img_path_list):
                label_name = str(idx).zfill(5) + '.png'
                writer.writerow([item,label_name])
                label_name_lst.append(label_name)

        print('00#############image path and label name Loaded from source!!!')

    
    else:
        img_path_list, label_name_lst = [], []
        with open (csv_save_path,'r') as f:
            reader = csv.reader(f,delimiter=',')
            header = next(reader)
            for item in reader:
                img_path_list.append(item[0])
                label_name_lst.append(item[1])
    
        print('00#############image path and label name Loaded from csv file!!!')


    return img_path_list, label_name_lst


def load_public_train_path(train_root,csv_save_path):
    label_name_lst = []
    img_path_list_train = glob(os.path.join(train_root,'*'))
    if not os.path.exists(csv_save_path):
        with open (csv_save_path,'w') as f:
            writer = csv.writer(f)
            header = ['img_path','label_name']
            writer.writerow(header)
            for idx, item in enumerate(img_path_list_train):
                label_name = str(idx).zfill(5) + '.png'
                label_name_lst.append(label_name)
                item_info = [item, label_name]
                writer.writerow(item_info)
    else:
        for idx, item in enumerate(img_path_list_train):
                label_name = str(idx).zfill(5) + '.png'
                label_name_lst.append(label_name)

    return img_path_list_train, label_name_lst
        

if __name__ == "__main__":
    root_dir_train = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/network_MIL_prediction_results/train/'
    public_2w_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/public_2w/V1/'
    csv_save_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/a_zhongshan_2wpublic_allzhongshan_2wpublic_imgPath.csv'
    img_path_list_train,label_name_lst = load_img_and_save_pos_path(root_dir_train,public_2w_path,csv_save_path)