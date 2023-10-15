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


def gather_align_EndoImg_Center(root_dir='/home/ubuntu/Data/database/中山', split=0.7):
    path_BagAnno = glob(os.path.join(root_dir, '*.xlsx'))[0]
    path_image = os.path.join(root_dir, '图像')
    # path_polypAnno = os.path.join(root_dir, '息肉')
    path_polypAnno = '/home/xiaoyuan/Data3/EndoGPT/database/Final/内部测试集'
    path_polypAnno2 = '/home/xiaoyuan/Data3/EndoGPT/database/Final/前瞻验证集'

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

    endo_patient_withInstLabel = glob(os.path.join(path_polypAnno, "*"))
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

    # separate patients with instance label to test set
    bag_data_all_tmp_forIdx = bag_data_all[:, 0].copy()
    bag_data_all_tmp_forIdx = np.array([os.path.basename(path) for path in bag_data_all_tmp_forIdx])

    endo_patient_withInstLabel = glob(os.path.join(path_polypAnno2, "*"))
    endo_patient_withInstLabel = np.array(endo_patient_withInstLabel)
    idx_test_withInstLabel2 = []
    for i in tqdm(range(endo_patient_withInstLabel.shape[0]), desc='Separate patients from Qian Zhan'):
        idx = endo_patient_withInstLabel[i].split('/')[-1].split('_')[0]
        search_idx = np.where(bag_data_all_tmp_forIdx == idx)[0]
        if len(search_idx) == 1:
            idx_test_withInstLabel2.append(search_idx)
        elif len(search_idx) > 1:
            raise
    idx_test_withInstLabel2 = np.array(idx_test_withInstLabel2).squeeze()
    idx_withoutInstLabel2 = np.setdiff1d(idx_withoutInstLabel, idx_test_withInstLabel2)

    num_patient_withoutInstLabel = len(idx_withoutInstLabel2)
    idx_train_test = np.random.choice(num_patient_withoutInstLabel, num_patient_withoutInstLabel, replace=False)
    idx_train = idx_withoutInstLabel2[idx_train_test[: int(split * num_patient)]]
    idx_test = idx_withoutInstLabel2[idx_train_test[int(split * num_patient):]]
    if len(idx_test_withInstLabel2) == 0:
        return bag_data_all[idx_train], bag_data_all[idx_test], bag_data_all[idx_test_withInstLabel], None
    return bag_data_all[idx_train], bag_data_all[idx_test], bag_data_all[idx_test_withInstLabel], bag_data_all[idx_test_withInstLabel2]


def gather_all_center(split=0.7):
    ZS_data_train, ZS_data_test, ZS_data_InternalTest_withInstLabel, ZS_data_QianZhanTest_withInstLabel = gather_align_EndoImg_Center(root_dir='/home/xiaoyuan/Data3/EndoGPT/database/中山', split=split)
    ZZ_data_train, ZZ_data_test, ZZ_data_InternalTest_withInstLabel, _ = gather_align_EndoImg_Center(root_dir='/home/xiaoyuan/Data3/EndoGPT/database/郑州', split=split)
    all_data_train = np.concatenate([ZS_data_train, ZZ_data_train], axis=0)
    all_data_test = np.concatenate([ZS_data_test, ZZ_data_test], axis=0)
    all_data_InternalTest_withInstLabel = np.concatenate([ZS_data_InternalTest_withInstLabel, ZZ_data_InternalTest_withInstLabel], axis=0)
    all_data_QianZhanTest_withInstLabel = ZS_data_QianZhanTest_withInstLabel
    return all_data_train, all_data_test, all_data_InternalTest_withInstLabel, all_data_QianZhanTest_withInstLabel


def gather_ZS_negBag(dir='/home/xiaoyuan/Data3/EndoGPT/database/ZS_N/N'):
    # gather new append negBag for evaluation
    data_all = []
    for i in os.listdir(dir):
        path_i = os.path.join(dir, i)
        data_all.append([path_i, '0'])
    data_all = np.array(data_all)
    return data_all


def gather_one_center(split=0.7, center=0):
    if center == 0:
        ZS_data_train, ZS_data_test, ZS_data_test_withInstLabel = gather_align_EndoImg_Center(root_dir='/home/ubuntu/Data/database/中山', split=split)
        return ZS_data_train, ZS_data_test, ZS_data_test_withInstLabel
    elif center == 1:
        XM_data_train, XM_data_test, XM_data_test_withInstLabel = gather_align_EndoImg_Center(root_dir='/home/ubuntu/Data/database/厦门', split=split)
        return XM_data_train, XM_data_test, XM_data_test_withInstLabel
    elif center == 2:
        ZZ_data_train, ZZ_data_test, ZZ_data_test_withInstLabel = gather_align_EndoImg_Center(root_dir='/home/ubuntu/Data/database/郑州', split=split)
        return ZZ_data_train, ZZ_data_test, ZZ_data_test_withInstLabel
    else:
        raise


def gather_external_withoutPathology(root_dir='/home/xiaoyuan/Data3/EndoGPT/database/外部/图像', num_usage=200, seed=0):
    data_bag_all = np.array(glob(os.path.join(root_dir, '*/')))

    print("[Attention] Assigning all External bags with POSITIVE Bag Label")
    data_bag_label = np.ones_like(data_bag_all)
    data_bag_all = np.stack([np.array(data_bag_all), data_bag_label], axis=1)

    # 2. Random select out num_usage of them
    np.random.seed(seed)
    idx_part = np.random.choice(data_bag_all.shape[0], num_usage, replace=False)
    data_bag_part = data_bag_all[idx_part]
    return data_bag_part


class Endo_img_MIL_all_center(torch.utils.data.Dataset):
    # @profile
    def __init__(self, ds, downsample=1.0, transform=None, return_bag=False):
        self.root_dir = ds
        self.transform = transform
        self.downsample = downsample
        self.return_bag = return_bag
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4738405, 0.30310306, 0.20231445],
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
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index == index)[0]

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
    def __init__(self, ds, downsample=1.0, transform=None, return_bag=False, replace='内部测试集'):
        self.root_dir = ds
        self.transform = transform
        self.downsample = downsample
        self.return_bag = return_bag
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
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
            # anno_path = patient_path.replace('图像', '息肉/图像')+'_息肉'
            anno_path = patient_path.replace('/郑州', '').replace('/中山', '').replace('/图像', '/Final/{}'.format(replace))
            if not os.path.exists(anno_path):
                raise
            pos_patches = glob(os.path.join(anno_path, "*/*.jpg"))
            neg_patches = glob(os.path.join(anno_path, "*.jpg"))
            pos_patches = [i.split('/')[-1] for i in pos_patches]
            if len(pos_patches) != 0:
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
            if len(neg_patches) != 0:
                for j, file_j in enumerate(glob(os.path.join(patient_path, "*.jpg"))):
                    self.all_patches.append(file_j)
                    self.patch_label.append(0)
                    self.patch_corresponding_slide_label.append(0)
                    self.patch_corresponding_slide_index.append(cnt_slide)
                    self.patch_corresponding_slide_name.append(patient_path.split('/')[-1])
                    cnt_patch = cnt_patch + 1
            if len(pos_patches) + len(neg_patches) == 0:
                raise
            cnt_slide = cnt_slide + 1
        self.num_patches = cnt_patch
        self.all_patches = np.array(self.all_patches)
        self.patch_label = np.array(self.patch_label)
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)

        # # verify each bag is a pos bag
        # for i in np.unique(self.patch_corresponding_slide_index):
        #     idx_same_slide = np.where(self.patch_corresponding_slide_index == i)[0]
        #     # print(idx_same_slide.shape)
        #     slide_patch_labels = self.patch_label[idx_same_slide]
        #     if slide_patch_labels.max() != 1:
        #         print("ERROR")

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


class Endo_img_MIL_external_withInstLabel(torch.utils.data.Dataset):
    # @profile
    def __init__(self, ds, downsample=1.0, transform=None, return_bag=False):
        self.root_dir = ds
        self.transform = transform
        self.downsample = downsample
        self.return_bag = return_bag
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
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
            anno_path = patient_path.replace('外部/图像', '外部中心数据精简化（前1000例）')
            # anno_path = patient_path
            if not os.path.exists(anno_path):
                raise
            pos_patches = glob(os.path.join(anno_path, "*/*.jpg"))
            neg_patches = glob(os.path.join(anno_path, "*.jpg"))
            pos_patches = [i.split('/')[-1] for i in pos_patches]
            if len(pos_patches) != 0:
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
            if len(neg_patches) != 0:
                for j, file_j in enumerate(glob(os.path.join(patient_path, "*.jpg"))):
                    self.all_patches.append(file_j)
                    self.patch_label.append(0)
                    self.patch_corresponding_slide_label.append(0)
                    self.patch_corresponding_slide_index.append(cnt_slide)
                    self.patch_corresponding_slide_name.append(patient_path.split('/')[-1])
                    cnt_patch = cnt_patch + 1
            if len(pos_patches) + len(neg_patches) == 0:
                raise
            if len(pos_patches) !=0 and len(neg_patches) != 0:
                raise
            cnt_slide = cnt_slide + 1
        self.num_patches = cnt_patch
        self.all_patches = np.array(self.all_patches)
        self.patch_label = np.array(self.patch_label)
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)

        # # verify each bag is a pos bag
        # for i in np.unique(self.patch_corresponding_slide_index):
        #     idx_same_slide = np.where(self.patch_corresponding_slide_index == i)[0]
        #     # print(idx_same_slide.shape)
        #     slide_patch_labels = self.patch_label[idx_same_slide]
        #     if slide_patch_labels.max() != 1:
        #         print("ERROR")

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


def cal_img_mean_std():
    ds_train, ds_test, ds_test_withInstLabel = gather_all_center()
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.4738405, 0.30310306, 0.20231445],
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
    # train_ds = Endo_img_MIL_all_center_CLIPFeat(split='train')
    # val_ds = Endo_img_MIL_all_center_CLIPFeat(split='test')
    # val_2_ds = Endo_img_MIL_all_center_CLIPFeat(split='test2')
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