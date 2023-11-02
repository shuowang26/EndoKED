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


def gather_align_EndoImg(root_dir='', split=0.7):
    raw_label = np.concatenate([
        np.array(pd.read_csv('')),
        np.array(pd.read_csv('')),
        # ...
    ], axis=1)
    clinical_info = pd.read_excel("").to_numpy()

    endo_patient_fanlin = glob("")
    endo_patient_all = endo_patient_fanlin
    endo_patient_all = np.array(endo_patient_all)

    # match img and label
    clip_label = []
    clip_path = []
    not_found_list = []
    overlap_found_list = []
    oversize_list = []
    for i in tqdm(range(raw_label.shape[0]), desc='Matching'):
        check_idx = raw_label[i, 0]
        search_idx = np.where(clinical_info[:, 1] == check_idx)[0]
        if len(search_idx) != 1:
            raise
        patient_name = clinical_info[search_idx[0], 2]

        find_flag = 0
        patient_dir = 0
        for patient_i in endo_patient_all:
            if patient_name == patient_i.split('/')[-1]:
                patient_dir = patient_i
                find_flag = find_flag + 1
        if find_flag == 0:
            not_found_list.append(patient_name)
        elif find_flag > 1:
            overlap_found_list.append(patient_name)
        else:
            sample_image = io.imread(glob(os.path.join(patient_dir, "*.JPG"))[0])
            clip_path.append(patient_dir)
            clip_label.append(raw_label[i])
    clip_path = np.array(clip_path)
    clip_label = np.array(clip_label)
    clip_data_all = np.concatenate([clip_path[:, None], clip_label], axis=1)

    num_patient = clip_data_all.shape[0]
    idx_train_test = np.random.choice(num_patient, num_patient, replace=False)
    idx_train = idx_train_test[: int(split * num_patient)]
    idx_test = idx_train_test[int(split * num_patient):]
    return clip_data_all[idx_train], clip_data_all[idx_test]


class Endo_img_MIL(torch.utils.data.Dataset):
    # @profile
    def __init__(self, ds, downsample=0.1, transform=None, return_bag=False, preload=False):
        self.root_dir = ds
        self.transform = transform
        self.downsample = downsample
        self.return_bag = return_bag
        self.preload = preload
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.31512642, 0.20470396, 0.13602576],
                                     std=[0.30684662, 0.21589851, 0.15230405])
            ])

        all_slides = ds

        # 1.1 down sample the slides
        print("================ Down sample Slide {} ================".format(downsample))
        np.random.shuffle(all_slides)
        all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        self.num_slides = len(all_slides)

        self.num_patches = 0
        for i in all_slides:
            self.num_patches = self.num_patches + len(glob(os.path.join(i[0], "*.JPG")))

        # 2.extract all available patches and build corresponding labels
        if self.preload:
            self.all_patches = np.zeros([self.num_patches, 576, 720, 3], dtype=np.uint8)
        else:
            self.all_patches = []
        self.patch_label = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        cnt_slide = 0
        cnt_patch = 0
        for i in tqdm(all_slides, ascii=True, desc='preload data'):
            patient_path = i[0]
            for j, file_j in enumerate(glob(os.path.join(patient_path, "*.JPG"))):
                if self.preload:
                    self.all_patches[cnt_patch, :, :, :] = io.imread(file_j)
                else:
                    self.all_patches.append(file_j)
                self.patch_label.append(0)
                self.patch_corresponding_slide_label.append(int(i[2]))
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(patient_path.split('/')[-1])
                cnt_patch = cnt_patch + 1
            cnt_slide = cnt_slide + 1
        self.num_patches = cnt_patch
        if not self.preload:
            self.all_patches = np.array(self.all_patches)
        self.patch_label = np.array(self.patch_label)
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)

        # # 4. sort patches into bag
        # self.all_bags = []
        # self.all_bags_label = []
        # for i in range(self.patch_corresponding_slide_index.max() + 1):
        #     idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index == i)[0]
        #     bag = self.all_patches[idx_patch_from_slide_i]
        #     self.all_bags.append(bag)
        #     patch_labels = self.patch_label[idx_patch_from_slide_i]
        #     slide_label = patch_labels.max()
        #     self.all_bags_label.append(slide_label)
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index==index)[0]
            if len(idx_patch_from_slide_i) > 100:
                idx_patch_from_slide_i = idx_patch_from_slide_i[:100]

            bag = self.all_patches[idx_patch_from_slide_i]
            bag_normed = np.zeros([bag.shape[0], 3, 512, 512], dtype=np.float32)
            for i in range(bag.shape[0]):
                if self.preload:
                    instance_img = bag[i]
                else:
                    instance_img = io.imread(bag[i])
                bag_normed[i, :, :, :] = self.transform(Image.fromarray(np.uint8(instance_img), 'RGB'))
            bag = bag_normed
            patch_labels = self.patch_label[idx_patch_from_slide_i]
            slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i].max()
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            return bag, [patch_labels, slide_label, slide_index, slide_name], index
        else:
            if self.preload:
                patch_image = self.all_patches[index]
            else:
                patch_image = io.imread(self.all_patches[index])
            patch_label = self.patch_label[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            # patch_image = patch_image[:, 35:35+512, 165:165+512]
            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches


class Endo_img_MIL_Feat(torch.utils.data.Dataset):
    def __init__(self, feat_dir = "./output_EndoImg_feat_512x512_CLIP(RN50)", train=True, return_bag=True):
        # Load saved CLIP feat
        self.train = train
        self.return_bag = return_bag

        if train:
            self.all_patches = np.load(os.path.join(feat_dir, "train_feats.npy"))
            self.patch_corresponding_slide_label = np.load(os.path.join(feat_dir, "train_corresponding_slide_label.npy"))
            self.patch_corresponding_slide_index = np.load(os.path.join(feat_dir, "train_corresponding_slide_index.npy"))
            self.patch_corresponding_slide_name = np.load(os.path.join(feat_dir, "train_corresponding_slide_name.npy"))
        else:
            self.all_patches = np.load(os.path.join(feat_dir, "test_feats.npy"))
            self.patch_corresponding_slide_label = np.load(os.path.join(feat_dir, "test_corresponding_slide_label.npy"))
            self.patch_corresponding_slide_index = np.load(os.path.join(feat_dir, "test_corresponding_slide_index.npy"))
            self.patch_corresponding_slide_name = np.load(os.path.join(feat_dir, "test_corresponding_slide_name.npy"))

        self.all_patches_label = np.zeros_like(self.patch_corresponding_slide_label)  # dummy
        print("Feat Loaded")

        # sort by slide index
        self.num_slides = self.patch_corresponding_slide_index.max() + 1
        self.num_patches = self.all_patches.shape[0]

        self.slide_feat_all = []
        self.slide_label_all = []
        self.slide_patch_label_all = []
        for i in range(self.num_slides):
            idx_from_same_slide = self.patch_corresponding_slide_index == i
            idx_from_same_slide = np.nonzero(idx_from_same_slide)[0]

            self.slide_feat_all.append(self.all_patches[idx_from_same_slide])
            if self.patch_corresponding_slide_label[idx_from_same_slide].max() != self.patch_corresponding_slide_label[idx_from_same_slide].min():
                raise
            self.slide_label_all.append(self.patch_corresponding_slide_label[idx_from_same_slide].max())
            self.slide_patch_label_all.append(np.zeros(idx_from_same_slide.shape[0]).astype(np.long))
        print("Feat Sorted")

    def __getitem__(self, index):
        if self.return_bag:
            return self.slide_feat_all[index], [self.slide_patch_label_all[index], self.slide_label_all[index]], index
        else:
            return self.all_patches[index], \
                [
                    self.all_patches_label[index],
                    self.patch_corresponding_slide_label[index],
                    self.patch_corresponding_slide_index[index],
                    self.patch_corresponding_slide_name[index]
                ], \
                index

    def __len__(self):
        if self.return_bag:
            return self.num_slides
        else:
            return self.num_patches


def cal_img_mean_std():
    ds_train, ds_test = gather_align_EndoImg()
    train_ds = Endo_img_MIL(ds=ds_train, transform=None, return_bag=False)
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
    # mean, std = cal_img_mean_std()
    # transform_data = transforms.Compose([
    #         transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.31512642, 0.20470396, 0.13602576], std=[0.30684662, 0.21589851, 0.15230405])])  # CAMELYON16_224x224_10x
    ds_train, ds_test = gather_align_EndoImg()
    train_ds = Endo_img_MIL(ds=ds_train, transform=None, return_bag=False)
    val_ds = Endo_img_MIL(ds=ds_test, transform=None, return_bag=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1,
                                               shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1,
                                             shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    patch_img_all = []
    for i, data in enumerate(tqdm(train_loader, desc='loading')):
        patch_img_all.append(data[0].shape)
        label_patch = data[1][0]
        label_bag = data[1][1]
        idx = data[-1]
    print("END")
