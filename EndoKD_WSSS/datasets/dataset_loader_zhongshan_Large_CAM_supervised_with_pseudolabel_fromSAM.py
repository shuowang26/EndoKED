import pandas as pd
from glob import glob
import numpy as np
from PIL import Image
import torch
import os
from torchvision import transforms as T
from . import transforms
# import transforms
import imageio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

def one_hot_encoding(labels, n_classes):
    result = np.eye(n_classes)[labels]
    return result

def load_img_label_info_from_csv(csv_dir,pos_num=2,):

    img_path_list = []
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
                img_path_list.append(path_name)
                one_hot_pos = one_hot_encoding(np.array([1]),2)
                # label.append(one_hot_pos)
                label.append(np.array([1.0]))
        elif 'gt0' in item_path:
            raw_idx = np.random.randint(len(sorted_reader.index)//2)
            # idx = sorted_reader.index[raw_idx]
            idx = sorted_reader.index[0]
            path_name = sorted_reader['0'][idx].replace('/home/ubuntu/Data/database/中山/','/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/中山/')
            img_path_list.append(path_name)
            # one_hot_neg = one_hot_encoding(np.array([0]),2)
            label.append(np.array([0.0]))
            # label.append(one_hot_neg)
    
    img_path_list = np.array(img_path_list)
    label = np.array(label)

    return img_path_list, label

def load_test_img_mask(root_dir):
    img_path_list = glob(os.path.join(root_dir,'Original/*'))
    label_path_list = glob(os.path.join(root_dir,'Ground Truth/*'))
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
        
        self.global_view1 = T.Compose([
            # T.RandomResizedCrop(224, scale=[0.4, 1], interpolation=Image.BICUBIC),
            # self.flip_and_color_jitter,
            # self.gaussian_blur(p=1.0),
            # self.normalize,
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

    def __transforms(self, image,seg_mask):
        img_box = None
        local_image = None
        if self.aug:
            if self.rescale_range:
                image, seg_mask = transforms.random_scaling(image,label=seg_mask, scale_range=self.rescale_range)

            if self.img_fliplr:
                image, seg_mask = transforms.random_fliplr(image,label=seg_mask,)
            if self.crop_size:
                image,seg_mask, img_box = transforms.random_crop_with_pseudo_labels_sam(image, label=seg_mask,crop_size=self.crop_size, mean_rgb=[123.675, 116.28, 103.53], ignore_index=self.ignore_index)
            local_image = self.local_view(Image.fromarray(image)).float()
            image = self.global_view1(Image.fromarray(image))
        
        image = np.array(image)
        image = transforms.normalize_img2(image)
        return image, local_image, seg_mask, img_box
    
    def __getitem__(self, index):
        img_item_path = self.img_path[index].replace('\n','')
        if self.type == 'val':
            img_name = img_item_path.split('Original/')[-1].replace('.png','')
            label_mask_path = self.label_path[index]
            seg_mask = np.asarray(Image.open(label_mask_path).convert('L')) == 255


        else:
            img_name = 'ZS00' + img_item_path.split('ZS00')[-1].replace('/','_').replace('.jpg','')
            label_mask_path = self.label_path + img_name + '.png' 
            seg_mask = np.asarray(Image.open(label_mask_path).convert('L')) == 255

        image = np.array(imageio.imread(img_item_path))
        cls_label = np.unique(seg_mask)[-1].astype(np.int16)
        cls_label = np.array([cls_label])

        pil_image = Image.fromarray(image)
        image, local_image, seg_mask, img_box = self.__transforms(image=image,seg_mask=seg_mask)
        # plt.imshow(seg_mask)
        # plt.axis('off')
        # plt.show()
        seg_mask= np.float32(seg_mask)
        image = np.transpose(image, (2, 0, 1))
        
        if self.aug:
            crops = []
            crops.append(image.astype(np.float64))
            crops.append(self.global_view2(pil_image).float())
            crops.append(local_image)
            # for _ in range(8):
            #     crops.append(self.local_view(pil_image))

            return  image, cls_label, seg_mask, img_box, crops
        else:
            return  img_name, image, seg_mask, cls_label

        
if __name__ == "__main__":

    root_dir_train = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/network_MIL_prediction_results/train/'
    root_dir_test = "/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/CVC-ClinicDB_datasets/CVC-ClinicDB_datasets/CVC-ClinicDB_PNG_datasets/"

    # img_path_list_train, label_train = load_img_label_info_from_csv(root_dir_train)
    label_train = '/home/jaye/Downloads/Endo_SAM/bbox_sam_with_preds_zhongshan0611_pseudolabels_preds_dice0.770/'
    with open ('/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/polyp_zhongshan_pos/pseudo_labels_img_path_from_decoder_trainedfrom_SAM_step6.txt','r') as f:

        img_path_list_train = f.readlines()
    img_path_list_test, mask_test_path = load_test_img_mask(root_dir_test)

    train_ds =  Endo_img_WSSS(img_path_list_train,label_train,aug=True,)
    train_loader = DataLoader(train_ds,batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False,
                              drop_last=True,)
    test_ds =  Endo_img_WSSS(img_path_list_test,mask_test_path,type='val')
    test_loader = DataLoader(test_ds,batch_size=1,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=False,
                              drop_last=True,)

        
    for data in train_loader:
        image, cls_label, seg_mask, img_box, crops = data
        plt.imshow(image[0].numpy().transpose(1,2,0))
        plt.axis('off')
        plt.title(f'{cls_label}')
        plt.show()
        plt.imshow(seg_mask[0].numpy())
        plt.axis('off')
        plt.show()
    for data in test_loader:
        img_name, image, seg_mask, cls_label = data
        plt.imshow(image[0].numpy().transpose(1,2,0))
        plt.axis('off')
        plt.show()
        plt.imshow(seg_mask[0].numpy())
        plt.axis('off')
        plt.show()