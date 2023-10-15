import os 
import pandas as pd
from glob import glob
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from a1_generate_all_pos_polyp_path_as_csv import load_img_and_save_pos_path
from torchvision import transforms as T


class Endo_img_WSSS(Dataset):
    def __init__(self,
                img_path,
                label_name_lst,
                **kwargs):
        super().__init__()

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.label_name_lst = label_name_lst
        self.img_path = img_path

    def normalize_img(self,img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] - mean[0]) / std[0]
        proc_img[..., 1] = (imgarr[..., 1] - mean[1]) / std[1]
        proc_img[..., 2] = (imgarr[..., 2] - mean[2]) / std[2]
        return proc_img


    def __len__(self):
        return len(self.img_path)
    

    def __getitem__(self, index):
        img_item_path = self.img_path[index]
        label_name = self.label_name_lst[index] 
        image = cv2.imread(img_item_path,cv2.IMREAD_COLOR)
        b,g,r = cv2.split(image)          #分别提取B、G、R通道
        image = cv2.merge([r,g,b]) 
        image_raw = image
        image = self.normalize(image)
        image = np.array(image)
        cls_label = np.array([1.0])
        data_info = [img_item_path,label_name]
        
        return  label_name, image, cls_label, image_raw, data_info

class test_dataset:
    def __init__(self, image_root, gt_root, testsize,normalize=True):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]

        if normalize:
            self.transform = T.Compose([
                T.Resize((self.testsize, self.testsize)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])
        else:
            self.transform = T.Compose([
                T.Resize((self.testsize, self.testsize)),
                T.ToTensor(),])
                
        self.gt_transform = T.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image_raw = np.array(image)
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image,image_raw, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    

        
if __name__ == "__main__":

    root_dir_train = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/network_MIL_prediction_results/train/'
    root_dir_test = "/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/CVC-ClinicDB_datasets/CVC-ClinicDB_datasets/CVC-ClinicDB_PNG_datasets/"
    public_2w_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/public_2w/V1/'
    csv_save_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/a_zhongshan_2wpublic_allzhongshan_2wpublic_imgPath.csv'
    img_path_list_train,label_name_lst = load_img_and_save_pos_path(root_dir_train,public_2w_path,csv_save_path)
    train_ds =  Endo_img_WSSS(img_path_list_train,label_name_lst)
    train_loader = DataLoader(train_ds,batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False,
                              drop_last=True,)
        
    print('done')
    for data in train_loader:
        img_name_train, image, cls_label,_,_ = data
        plt.imshow(image[0].numpy())
        plt.axis('off')
        plt.title(f'{cls_label}')
        plt.show()