import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt 
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def denormalize_img(imgs=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:,0,:,:] = imgs[:,0,:,:] * std[0] + mean[0]
    _imgs[:,1,:,:] = imgs[:,1,:,:] * std[1] + mean[1]
    _imgs[:,2,:,:] = imgs[:,2,:,:] * std[2] + mean[2]

    return _imgs

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_list, label_lst, trainsize, augmentations,normalize):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.normalize = normalize
        print(self.augmentations)
        print(f'normalization is {self.normalize}')
        self.images = image_list
        self.labels = label_lst

        # self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        # self.filter_files()
        self.size = len(self.images)
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            if self.normalize:
                print('normalize')
                self.img_transform = transforms.Compose([
                    transforms.Resize((self.trainsize, self.trainsize)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
            else:
                print('no normalize')
                self.img_transform = transforms.Compose([
                    transforms.Resize((self.trainsize, self.trainsize)),
                    transforms.ToTensor(),])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image_item_path = self.images[index]
        image = self.rgb_loader(image_item_path)

        label_mask_path = self.labels[index]
        gt = self.binary_loader(label_mask_path)
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        # seed = np.random.randint(100) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True, augmentation=False,normalize=True):
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation,normalize)
    train_sampler = DistributedSampler(dataset, shuffle=shuffle)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize,normalize=True):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),])
                
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == '__main__':

    test_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/dataset/TestDataset/'
    dataset = 'test'
    data_path = os.path.join(test_path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    num1 = len(os.listdir(gt_root))
    tb_path = '/data/PROJECTS/Endo_GPT/reference_codes/Polyp-PVT-main/xxx_tb'
    writer = SummaryWriter(tb_path)
    test_loader = test_dataset(image_root, gt_root, 352)
    images = []
    gts = []
    resize = transforms.Resize((352, 352))
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        image = denormalize_img(image)
        if i < 24:
            images.append(image[0])
            gt = resize(gt)
            gts.append(torch.tensor(np.array(gt)[None,...]))
        # plt.subplot(121)
        # plt.imshow(grid_image.permute(1,2,0))
        # plt.subplot(122)
        # plt.imshow(gt)
        # plt.show()
        # plt.subplot(122)
        # plt.imshow(grid_image.permute(1,2,0).cpu())
        # plt.show()
        if i == 23:
            images = torch.stack(images,dim=0)
            gts = torch.stack(gts,dim=0)
            grid_image  = torchvision.utils.make_grid(images,6)
            grid_gt  = torchvision.utils.make_grid(gts.clone(),6)
            writer.add_image(f"val/i{dataset}_mages",grid_image,global_step = i)
            writer.add_image(f"val/i{dataset}_gt",grid_gt,global_step = i)