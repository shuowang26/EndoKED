
###### zhongshan 阳性样本加上阈值； 加上两万例阳性样本； 加上所有阴性样本中山

import pandas as pd
from glob import glob
import numpy as np
from PIL import Image
import torch
import os
import cv2
import json
from torchvision import transforms as T
import csv
from . import transforms
# import transforms
import imageio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


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


def load_img_info_from_csv_public_2w(csv_dir,public_2w_path,threshold=0.8,pos_num=5,):

    pub_img_pos_lst, pub_label_pos_lst, pub_img_neg_lst, pub_label_neg_lst = load_img_path_from_2w(public_2w_path)
    pos_img_path_list = []
    neg_img_path_list = []
    pos_label = []
    neg_label = []
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
                    pos_img_path_list.append(path_name)
                    # one_hot_pos = one_hot_encoding(np.array([1]),2)
                    # label.append(one_hot_pos)
                    pos_label.append(np.array([1.0]))
            
        elif 'gt0' in item_path:
            # raw_idx = np.random.randint(len(sorted_reader.index)//2)
            # idx = sorted_reader.index[raw_idx]
            # idx = sorted_reader.index[0]
            for idx in range(len(sorted_reader.index)//20):
                path_name = sorted_reader['0'][idx].replace('/home/ubuntu/Data/database/中山/','/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/中山/')
                neg_img_path_list.append(path_name)
                # one_hot_neg = one_hot_encoding(np.array([0]),2)
                neg_label.append(np.array([0.0]))
            # label.append(one_hot_neg)
                
    # img_path_list = pos_img_path_list + pub_img_pos_lst
    # label = pos_label + pub_label_pos_lst
    img_path_list = pos_img_path_list + neg_img_path_list + pub_img_pos_lst + pub_img_neg_lst
    label = pos_label + neg_label + pub_label_pos_lst + pub_label_neg_lst
    # img_path_list = pub_img_pos_lst + pub_img_neg_lst
    # label = pub_label_pos_lst + pub_label_neg_lst
    label = np.array(label)

    return img_path_list, label

def load_test_img_mask(root_dir):
    img_path_list = glob(os.path.join(root_dir,'images/*'))
    label_path_list = glob(os.path.join(root_dir,'masks/*'))
    return img_path_list, label_path_list

class Endo_img_WSSS(Dataset):
    def __init__(self,
                img_path,
                label_path,
                type='train',
                resize_range=[512, 640],
                rescale_range=[0.6, 1.0],
                crop_size=448,
                img_fliplr=True,
                ignore_index=255,
                num_classes=2,
                aug=False,
                **kwargs):
        super().__init__()

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.local_crop_size = 96
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.type = type
        self.label_path = label_path
        self.img_path = img_path

        #transforms
        self.gaussian_blur = transforms.GaussianBlur
        self.solarization = transforms.Solarization(p=0.2)
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

        self.resize = T.Compose([
            T.Resize((448, 448)),
            # T.RandomCrop((self.crop_size,self.crop_size)),
            T.ToTensor(),
            # T.RandomVerticalFlip(p=0.5),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.gt_resize = T.Compose([
            T.Resize((448, 448)),
            # T.RandomCrop((self.crop_size,self.crop_size)),
            T.ToTensor(),
            ])
        
        self.local_view = T.Compose([
            # T.RandomResizedCrop(self.local_crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
        ])
        self.global_view2 = T.Compose([
            T.RandomResizedCrop(self.crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.1),
            self.solarization,
            self.normalize,
        ])
        self.local_view = T.Compose([
            # T.RandomResizedCrop(self.local_crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.5),
            self.normalize,
        ])

    def __len__(self):
        return len(self.img_path)

    def __transforms(self, image,mask):
        img_box = 1.0
        local_image = None
        if self.aug:

            image = np.array(self.resize(Image.fromarray(image)))
            gt = np.array(self.gt_resize(Image.fromarray(mask)))
            image = image.transpose(1,2,0)
            local_image = self.local_view(Image.fromarray(image.astype(np.uint8))).float()
            image = np.transpose(image, (2, 0, 1))
            return image, gt[0],local_image, img_box

        else:
	    #image = np.transpose(image, (2, 0, 1)) /255 
            #image = np.transpose(image, (2, 0, 1))
            image = self.normalize(image)
            image = np.array(image)
            return image, mask, local_image, img_box


    
    def __getitem__(self, index):
        img_item_path = self.img_path[index]
        if self.type == 'val':
            img_name = img_item_path.split('images/')[-1].replace('.png','')

        # image = np.array(imageio.imread(img_item_path))
        image = cv2.imread(img_item_path,cv2.IMREAD_COLOR)
        b,g,r = cv2.split(image)  #分别提取B、G、R通道
        image = cv2.merge([r,g,b]) 
        label_mask_path = self.label_path[index]
        seg_mask = np.asarray(Image.open(label_mask_path).convert('L')) == 255
        pil_image = Image.fromarray(image)


        image, seg_mask, local_image, img_box = self.__transforms(image=image,mask=seg_mask)

        cls_label = np.unique(seg_mask)[-1].astype(np.int16)
        cls_label = np.array(cls_label)
        # cls_label = np.eye(2)[cls_label]
        
        if self.aug:
            crops = []
            crops.append(image.astype(np.float64))
            crops.append(self.global_view2(pil_image).float())
            crops.append(local_image)

            return  image, cls_label, seg_mask, img_box, crops
        else:
            return  img_name, image, seg_mask, cls_label
            
if __name__ == "__main__":

    label_root_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/0623_Preds_seeds/'
    img_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/a_zhongshan_2wpublic_all/filtered_after_Preds_zhongshan_2wpublic_imgPath.csv'
    img_path_list, label_name_lst = [], []

    with open (img_path,'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for item in reader:
            img_path_list.append(item[0])
            label_path = label_root_path + item[1]
            label_name_lst.append(label_path)

    train_ds =  Endo_img_WSSS(img_path_list, label_name_lst,aug=True,)
    train_loader = DataLoader(train_ds,batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False,
                              drop_last=True,)
    test_ds =  Endo_img_WSSS(img_path_list, label_name_lst,type='val')
    test_loader = DataLoader(test_ds,batch_size=1,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=False,
                              drop_last=True,)

        
    # for data in train_loader:
    #     image, cls_label, mask, img_box, crops = data
    #     plt.subplot(121)
    #     plt.imshow(image[0].numpy().transpose(1,2,0))
    #     plt.axis('off')
    #     plt.title(f'{cls_label}')
    #     plt.subplot(122)
    #     plt.imshow(mask[0])
    #     plt.show()
    for data in test_loader:
        img_name, image, seg_mask, cls_label = data
        plt.subplot(121)
        plt.imshow(image[0].numpy().transpose(1,2,0))
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(seg_mask[0].numpy())
        plt.axis('off')
        plt.show()