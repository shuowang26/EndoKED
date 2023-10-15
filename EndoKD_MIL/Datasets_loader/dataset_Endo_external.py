import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image
import os
from glob import glob
from skimage import io
from tqdm import tqdm
import pandas as pd


def gather_align_EndoImg_External():
    cvc_file_path = glob('/root/Data2/external_ds/public_ds/CVC-ClinicDB_datasets/CVC-ClinicDB_datasets/CVC-ClinicDB_PNG_datasets/Original/*.png')
    cvc_label = np.ones(len(cvc_file_path))

    kvasir_file_path = glob("/root/Data2/external_ds/public_ds/kvasir-dataset-v2/normal*/*.jpg")
    kvasir_label = np.zeros(len(kvasir_file_path))
    return kvasir_file_path, kvasir_label

    file_path = cvc_file_path+kvasir_file_path
    label = np.concatenate([cvc_label, kvasir_label])
    return file_path, label


class Endo_img_External(torch.utils.data.Dataset):
    # @profile
    def __init__(self, transform=None):
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.46547693, 0.2973105, 0.19359106],
                                     std=[0.26098314, 0.19691008, 0.14406867])
            ])
        self.all_patches, self.patch_label = gather_align_EndoImg_External()
        self.num_patches = len(self.all_patches)

        print("")

    def __getitem__(self, index):
        patch_image = io.imread(self.all_patches[index])
        patch_label = self.patch_label[index]
        patch_corresponding_slide_label = 0
        patch_corresponding_slide_index = 0
        patch_corresponding_slide_name = 0

        patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
        # patch_image = patch_image[:, 35:35+512, 165:165+512]
        return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                             patch_corresponding_slide_name], index

    def __len__(self):
        return self.num_patches


def cal_img_mean_std():
    train_ds = Endo_img_External(transform=None)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128,
                                               shuffle=False, num_workers=6, drop_last=True, pin_memory=True)
    print("Length of dataset: {}".format(len(train_ds)))
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data in tqdm(train_loader, desc="Calculating Mean and Std"):
        img = data[0]
        for d in range(3):
            mean[d] += img[:, d, :, :].mean()
            std[d] += img[:, d, :, :].std()
    mean.div_(len(train_ds))
    std.div_(len(train_ds))
    mean = list(mean.numpy()*128)
    std = list(std.numpy()*128)
    print("Mean: {}".format(mean))
    print("Std: {}".format(std))
    return mean, std


if __name__ == '__main__':
    mean, std = cal_img_mean_std()

    val_ds = Endo_img_External(transform=None)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1,
                                             shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    patch_img_all = []
    for i, data in enumerate(tqdm(val_loader, desc='loading')):
        patch_img_all.append(data[0].shape)
        label_patch = data[1][0]
        label_bag = data[1][1]
        idx = data[-1]
    print("END")