import os
import os.path as osp
from utils.transform_multi import *
from torch.utils.data import Dataset
from torchvision import transforms
import csv
from glob import glob


def load_data_from_zhnogshan():
    root = './data/poly_detection'
    img_path = f'{root}/images/*'
    mask_path = f'{root}/masks/*'

    img_path_list, label_path_lst = sorted(glob(img_path)), sorted(glob(mask_path))

    return img_path_list, label_path_lst

class PolypDataset(Dataset):
    def __init__(self, args, data_dir, mode='train', transform=None):
        super(PolypDataset, self).__init__()
        data_path = data_dir
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        if args.dataset == "Train_on_ZhongshanandKvasirandDB" and mode=='train':
            zhong_img_lst,zhong_mask_lst=load_data_from_zhnogshan()
            self.images += zhong_img_lst
            self.gts += zhong_mask_lst

        if args.dataset == "Train_on_Zhongshan" and mode=='train':
            zhong_img_lst,zhong_mask_lst=load_data_from_zhnogshan()
            self.images = zhong_img_lst
            self.gts = zhong_mask_lst

        if transform is None:
            if mode == 'train':
               transform = transforms.Compose([
                   Resize((256, 256)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   RandomCrop((224, 224)),
                   ToTensor(),
               ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((224, 224)),
                   ToTensor(),
               ])
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.images[index]
        gt_path = self.gts[index]
        name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        data = {'image': img, 'label': gt}
        if self.transform:
            data = self.transform(data)
        data['name'] = name
        return data

    def __len__(self):
        return len(self.images)
    

class PolygenDataset(Dataset):
    def __init__(self, args, data_dir, mode='train', transform=None):
        super(PolygenDataset, self).__init__()
        data_path = data_dir
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)


        if transform is None:
            if mode == 'train':
               transform = transforms.Compose([
                   Resize((256, 256)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   RandomCrop((224, 224)),
                   ToTensor(),
               ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((224, 224)),
                   ToTensor(),
               ])
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.images[index]
        gt_path = self.gts[index]
        name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        data = {'image': img, 'label': gt}
        if self.transform:
            data = self.transform(data)
        data['name'] = name
        return data

    def __len__(self):
        return len(self.images)

