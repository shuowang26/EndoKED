import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image
import os
from glob import glob
from skimage import io
from tqdm import tqdm
import pandas as pd


def gather_align_EndoImg_Center(root_dir='/home/xiaoyuan/Desktop/EndoGPT_Server/database/中山', split=0.7):
    path_BagAnno = glob(os.path.join(root_dir, '*.xlsx'))[0]
    path_image = os.path.join(root_dir, '图像')
    path_polypAnno = os.path.join(root_dir, '息肉')

    df_ = pd.read_excel(path_BagAnno)
    clinical_info = df_[['检查序号', '息肉备注']].to_numpy()

    endo_patient_all = glob(os.path.join(path_image, "*"))
    endo_patient_all = np.array(endo_patient_all)

    # match img and label
    bag_path = []
    bag_label = []
    not_found_list = []
    overlap_list = []
    for i in tqdm(range(endo_patient_all.shape[0]), desc='Matching'):
        check_idx = endo_patient_all[i].split('/')[-1]
        search_idx = np.where(clinical_info[:, 0] == check_idx)[0]
        if len(search_idx) > 1:
            overlap_list.append(endo_patient_all[i])
        elif len(search_idx) == 0:
            not_found_list.append(endo_patient_all[i])
        else:
            bag_path.append(endo_patient_all[i])
            bag_label.append(clinical_info[search_idx, 1].astype(int))

    bag_path = np.array(bag_path)
    bag_label = np.array(bag_label)
    bag_data_all = np.concatenate([bag_path[:, None], bag_label], axis=1)
    num_patient = bag_data_all.shape[0]

    # separate patients with instance label to test set
    bag_data_all_tmp_forIdx = bag_data_all[:, 0].copy()
    bag_data_all_tmp_forIdx = np.array([os.path.basename(path) for path in bag_data_all_tmp_forIdx])

    endo_patient_withInstLabel = glob(os.path.join(path_polypAnno, "图像/*"))
    endo_patient_withInstLabel = np.array(endo_patient_withInstLabel)
    idx_test_withInstLabel = []
    for i in tqdm(range(endo_patient_withInstLabel.shape[0]), desc='Separate patients with instance label'):
        idx = endo_patient_withInstLabel[i].split('/')[-1].split('_')[0]
        search_idx = np.where(bag_data_all_tmp_forIdx == idx)[0]
        if len(search_idx) == 1:
            idx_test_withInstLabel.append(search_idx)
        elif len(search_idx) > 1:
            raise
    idx_test_withInstLabel = np.array(idx_test_withInstLabel).squeeze()
    idx_withoutInstLabel = np.setdiff1d(np.arange(num_patient), idx_test_withInstLabel)
    num_patient_withoutInstLabel = len(idx_withoutInstLabel)

    idx_train_test = np.random.choice(num_patient_withoutInstLabel, num_patient_withoutInstLabel, replace=False)
    idx_train = idx_withoutInstLabel[idx_train_test[: int(split * num_patient)]]
    idx_test = idx_withoutInstLabel[idx_train_test[int(split * num_patient):]]
    return bag_data_all[idx_train], bag_data_all[idx_test], bag_data_all[idx_test_withInstLabel]


def gather_all_center(split=0.7):
    ZS_data_train, ZS_data_test, ZS_data_test_withInstLabel = gather_align_EndoImg_Center(root_dir='/home/xiaoyuan/Data3/EndoGPT/database/中山', split=split)
    XM_data_train, XM_data_test, XM_data_test_withInstLabel = gather_align_EndoImg_Center(root_dir='/home/xiaoyuan/Data3/EndoGPT/database/厦门', split=split)
    ZZ_data_train, ZZ_data_test, ZZ_data_test_withInstLabel = gather_align_EndoImg_Center(root_dir='/home/xiaoyuan/Data3/EndoGPT/database/郑州', split=split)
    all_data_train = np.concatenate([ZS_data_train, XM_data_train, ZZ_data_train], axis=0)
    all_data_test = np.concatenate([ZS_data_test, XM_data_test, ZZ_data_test], axis=0)
    all_data_test_withInstLabel = np.concatenate([ZS_data_test_withInstLabel, XM_data_test_withInstLabel, ZZ_data_test_withInstLabel], axis=0)
    return all_data_train, all_data_test, all_data_test_withInstLabel


def gather_one_center(split=0.7, center=0):
    if center == 0:
        ZS_data_train, ZS_data_test, ZS_data_test_withInstLabel = gather_align_EndoImg_Center(root_dir='/home/xiaoyuan/Data3/EndoGPT/database/中山', split=split)
        return ZS_data_train, ZS_data_test, ZS_data_test_withInstLabel
    elif center == 1:
        XM_data_train, XM_data_test, XM_data_test_withInstLabel = gather_align_EndoImg_Center(root_dir='/home/xiaoyuan/Data3/EndoGPT/database/厦门', split=split)
        return XM_data_train, XM_data_test, XM_data_test_withInstLabel
    elif center == 2:
        ZZ_data_train, ZZ_data_test, ZZ_data_test_withInstLabel = gather_align_EndoImg_Center(root_dir='/home/xiaoyuan/Data3/EndoGPT/database/郑州', split=split)
        return ZZ_data_train, ZZ_data_test, ZZ_data_test_withInstLabel
    else:
        raise


class Endo_img_MIL_all_center(torch.utils.data.Dataset):
    # @profile
    def __init__(self, ds, downsample=1.0, transform=None, return_bag=False):
        self.root_dir = ds
        self.transform = transform
        self.downsample = downsample
        self.return_bag = return_bag
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47403994, 0.29777905, 0.18895507],
                                     std=[0.2768913, 0.2088029, 0.16106644])
            ])

        all_slides = ds

        # 1.1 down sample the slides
        print("================ Down sample Slide {} ================".format(downsample))
        np.random.shuffle(all_slides)
        all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        self.num_slides = len(all_slides)

        # 2.extract all available patches and build corresponding labels
        self.all_patches = []
        self.patch_label = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        cnt_slide = 0
        cnt_patch = 0
        for i in tqdm(all_slides, ascii=True, desc='Scanning all bags'):
            patient_path = i[0]
            if len(glob(os.path.join(patient_path, "*.jpg"))) <= 1:
                continue
            else:
                for j, file_j in enumerate(glob(os.path.join(patient_path, "*.jpg"))):
                    self.all_patches.append(file_j)
                    self.patch_label.append(0)
                    self.patch_corresponding_slide_label.append(int(i[1]))
                    self.patch_corresponding_slide_index.append(cnt_slide)
                    self.patch_corresponding_slide_name.append(patient_path.split('/')[-1])
                    cnt_patch = cnt_patch + 1
                cnt_slide = cnt_slide + 1
        self.num_patches = cnt_patch
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

            bag = self.all_patches[idx_patch_from_slide_i]
            bag_normed = np.zeros([bag.shape[0], 3, 224, 224], dtype=np.float32)
            for i in range(bag.shape[0]):
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
            patch_image = io.imread(self.all_patches[index])
            patch_label = self.patch_label[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches


class Endo_img_MIL_all_center_withInstLabel(torch.utils.data.Dataset):
    # @profile
    def __init__(self, ds, downsample=1.0, transform=None, return_bag=False):
        self.root_dir = ds
        self.transform = transform
        self.downsample = downsample
        self.return_bag = return_bag
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47403994, 0.29777905, 0.18895507],
                                     std=[0.2768913, 0.2088029, 0.16106644])
            ])

        all_slides = ds

        # 1.1 down sample the slides
        print("================ Down sample Slide {} ================".format(downsample))
        np.random.shuffle(all_slides)
        all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        self.num_slides = len(all_slides)

        # 2.extract all available patches and build corresponding labels
        self.all_patches = []
        self.patch_label = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        cnt_slide = 0
        cnt_patch = 0
        for i in tqdm(all_slides, ascii=True, desc='Scanning all bags'):
            patient_path = i[0]
            anno_path = patient_path.replace('图像', '息肉/图像')+'_息肉'
            if not os.path.exists(anno_path):
                raise
            pos_patches = os.listdir(anno_path)
            if len(pos_patches) == 0:
                raise
            for j, file_j in enumerate(glob(os.path.join(patient_path, "*.jpg"))):
                self.all_patches.append(file_j)
                if file_j.split('/')[-1] in pos_patches:
                    self.patch_label.append(1)
                else:
                    self.patch_label.append(0)
                self.patch_corresponding_slide_label.append(1)
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(patient_path.split('/')[-1])
                cnt_patch = cnt_patch + 1
            cnt_slide = cnt_slide + 1
        self.num_patches = cnt_patch
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

            bag = self.all_patches[idx_patch_from_slide_i]
            bag_normed = np.zeros([bag.shape[0], 3, 224, 224], dtype=np.float32)
            for i in range(bag.shape[0]):
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
            patch_image = io.imread(self.all_patches[index])
            patch_label = self.patch_label[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches


class Endo_img_MIL_all_center_CLIPFeat(torch.utils.data.Dataset):
    def __init__(self, feat_dir="./output_EndoImg3Center_feat_224x224_CLIP(RN50)", split='train', return_bag=True):
        # Load saved CLIP feat
        self.split = split
        self.return_bag = return_bag

        if self.split == 'train':
            self.all_patches = np.load(os.path.join(feat_dir, "train_feats.npy"))
            self.all_patches_name = np.load(os.path.join(feat_dir, "train_patch_name.npy"))
            self.patch_corresponding_slide_label = np.load(os.path.join(feat_dir, "train_corresponding_slide_label.npy"))
            self.patch_corresponding_slide_index = np.load(os.path.join(feat_dir, "train_corresponding_slide_index.npy"))
            self.patch_corresponding_slide_name = np.load(os.path.join(feat_dir, "train_corresponding_slide_name.npy"))
            self.all_patches_label = np.zeros_like(self.patch_corresponding_slide_label)  # dummy
        elif self.split == 'test':
            self.all_patches = np.load(os.path.join(feat_dir, "test_feats.npy"))
            self.all_patches_name = np.load(os.path.join(feat_dir, "test_patch_name.npy"))
            self.patch_corresponding_slide_label = np.load(os.path.join(feat_dir, "test_corresponding_slide_label.npy"))
            self.patch_corresponding_slide_index = np.load(os.path.join(feat_dir, "test_corresponding_slide_index.npy"))
            self.patch_corresponding_slide_name = np.load(os.path.join(feat_dir, "test_corresponding_slide_name.npy"))
            self.all_patches_label = np.zeros_like(self.patch_corresponding_slide_label)  # dummy
        elif self.split == 'test2':
            self.all_patches = np.load(os.path.join(feat_dir, "testWithInstLabel_feats.npy"))
            self.all_patches_name = np.load(os.path.join(feat_dir, "testWithInstLabel_patch_name.npy"))
            self.patch_corresponding_slide_label = np.load(os.path.join(feat_dir, "testWithInstLabel_corresponding_slide_label.npy"))
            self.patch_corresponding_slide_index = np.load(os.path.join(feat_dir, "testWithInstLabel_corresponding_slide_index.npy"))
            self.patch_corresponding_slide_name = np.load(os.path.join(feat_dir, "testWithInstLabel_corresponding_slide_name.npy"))
            self.all_patches_label = np.load(os.path.join(feat_dir, "testWithInstLabel_patch_label.npy"))
        else:
            raise

        print("Feat Loaded")

        # sort by slide index
        self.num_slides = len(np.unique(self.patch_corresponding_slide_index))
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
            self.slide_patch_label_all.append(self.all_patches_label[idx_from_same_slide].astype(np.compat.long))
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
    ds_train, ds_test, ds_test_withInstLabel = gather_all_center()
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.47403994, 0.29777905, 0.18895507],
        #                      std=[0.2768913, 0.2088029, 0.16106644])
    ])
    train_ds = Endo_img_MIL_all_center(ds=ds_train, downsample=1.0, transform=transform, return_bag=False)
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
    train_ds = Endo_img_MIL_all_center_CLIPFeat(split='train')
    val_ds = Endo_img_MIL_all_center_CLIPFeat(split='test')
    val_2_ds = Endo_img_MIL_all_center_CLIPFeat(split='test2')
    # mean, std = cal_img_mean_std()
    # transform_data = transforms.Compose([
    #         transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.47403994, 0.29777905, 0.18895507], std=[0.2768913, 0.2088029, 0.16106644])])
    ds_train, ds_test, ds_test_withInstLabel = gather_all_center()
    train_ds = Endo_img_MIL_all_center(ds=ds_train, transform=None, return_bag=False)
    val_ds = Endo_img_MIL_all_center(ds=ds_test, transform=None, return_bag=False)
    val_ds_withInstLabel = Endo_img_MIL_all_center_withInstLabel(ds=ds_test_withInstLabel, transform=None, return_bag=False)
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