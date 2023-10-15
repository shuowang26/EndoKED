import pandas as pd
from glob import glob
import numpy as np
from PIL import Image

from torchvision import transforms as T
from . import transforms
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

class Endo_img_WSSS(Dataset):
    def __init__(self,
                img_path,
                label_path,
                **kwargs):
        super().__init__()

        self.label_path = label_path
        self.img_path = img_path


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_item_path = self.img_path[index].replace('\n','')
        img_name_train = img_item_path 
        image = cv2.imread(img_item_path,cv2.IMREAD_COLOR)
        b,g,r = cv2.split(image)          #分别提取B、G、R通道
        image = cv2.merge([r,g,b]) 
        image = transforms.normalize_img2(image)
        image = np.transpose(image, (2, 0, 1))
        cls_label = self.label_path[index]
        
        return  img_name_train, image, cls_label

        
if __name__ == "__main__":

    with open ('/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/polyp_zhongshan_pos/selected_polyp_path.txt','r') as f:

        img_path_list_train = f.readlines()
    label_train = np.ones((len(img_path_list_train),1))

    train_ds =  Endo_img_WSSS(img_path_list_train,label_train)
    train_loader = DataLoader(train_ds,batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False,
                              drop_last=True,)
        
    for data in train_loader:
        img_name_train, image, cls_label = data
        plt.imshow(image[0].numpy().transpose(1,2,0))
        plt.axis('off')
        plt.title(f'{cls_label}')
        plt.show()
    # for data in test_loader:
    #     img_name, image, seg_mask, cls_label = data
    #     plt.imshow(image[0].numpy().transpose(1,2,0))
    #     plt.axis('off')
    #     plt.show()
    #     plt.imshow(seg_mask[0].numpy())
    #     plt.axis('off')
    #     plt.show()