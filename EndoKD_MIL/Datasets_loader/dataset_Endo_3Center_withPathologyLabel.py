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
from copy import deepcopy


def gather_align_EndoImg_Center_forPathology(root_dir='',
                                             pred_dir=None,
                                             labeled_dir=''):
    path_BagAnno = glob(os.path.join(root_dir, '*.xlsx'))[0]
    path_image = os.path.join(root_dir, '图像')
    path_polypAnno = labeled_dir

    df_ = pd.read_excel(path_BagAnno, na_values=["", " "], keep_default_na=False)
    df_.fillna(0, inplace=True)
    clinical_info = df_[['检查序号', '息肉备注', '标签1', '标签3', '标签4', '标签5']].to_numpy()

    endo_patient_all = glob(os.path.join(path_image, "*"))
    endo_patient_all = np.array(endo_patient_all)

    # 0. match img and label
    bag_path = []
    bag_label = []
    pathology_label = []
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
            pathology_label.append(clinical_info[search_idx, 2:6].squeeze().astype(int))

    bag_path = np.array(bag_path)
    bag_label = np.array(bag_label)
    pathology_label = np.array(pathology_label)
    bag_data_all = np.concatenate([bag_path[:, None], bag_label, pathology_label], axis=1)
    num_patient = bag_data_all.shape[0]
    print("Number of matched patients: {}".format(num_patient))

    # 1.1 separate patients with instance label to independent test set
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
    num_patient_withoutInstLabel = len(idx_withoutInstLabel)
    bag_data_all_withoutInstLabel = bag_data_all[idx_withoutInstLabel]
    bag_data_all_withInstLabel = bag_data_all[idx_test_withInstLabel]
    print("Number of patients without GT instance label: {}".format(bag_data_all_withoutInstLabel.shape))
    print("Number of patients with GT instance label: {}".format(bag_data_all_withInstLabel.shape))

    # 1.2 index ground truth instances for patients with instance label
    data_independent_test = []
    for i in tqdm(range(len(bag_data_all_withInstLabel)), ascii=True, desc='Scanning bags with GT instance label'):
        patient_path = bag_data_all_withInstLabel[i, 0]
        patient_pathologyLabel = bag_data_all_withInstLabel[i, 2:6].astype(int)
        anno_path = patient_path.replace('图像', '息肉/{}'.format(labeled_dir.split('/')[-1])) + '_息肉'
        if not os.path.exists(anno_path):
            raise
        for pathology_type, pathology_type_code in zip(['病理类型_0', '病理类型_1', '病理类型_2', '病理类型_3'],
                                                       [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]):
            pos_patches = glob(os.path.join(anno_path, pathology_type, '*.jpg'))
            if len(pos_patches) == 0:
                pass
            else:
                # data_independent_test.append([
                #     patient_path.split('/')[-1],
                #     np.concatenate([np.array(pos_patches).astype(object)[:, None], np.ones(len(pos_patches), dtype=float)[:, None]], axis=1),
                #     pathology_type_code
                # ])
                for pos_patches_i in pos_patches:
                    data_independent_test.append([
                        patient_path.split('/')[-1],
                        np.concatenate([np.array([pos_patches_i]).astype(object)[:, None], np.ones(1, dtype=float)[:, None]], axis=1),
                        pathology_type_code
                    ])

    # 2.1 filter invalid patients: (1) No polyp patients; (2) multiple pathology label == 1; (3) polyp label == 1 but pathology label == 0
    bag_data_all_withoutInstLabel_tmpForIdx = bag_data_all_withoutInstLabel[:, 1:].astype(int).copy()
    idx_valid = []
    for i in tqdm(range(bag_data_all_withoutInstLabel_tmpForIdx.shape[0]), desc='Filter invalid patients'):
        if bag_data_all_withoutInstLabel_tmpForIdx[i, 0] == 0:
            continue
        if bag_data_all_withoutInstLabel_tmpForIdx[i, 0] == 1 and np.sum(bag_data_all_withoutInstLabel_tmpForIdx[i, 1:]) == 0:
            continue
        elif np.sum(bag_data_all_withoutInstLabel_tmpForIdx[i, 1:]) > 1:
            continue
        else:
            idx_valid.append(i) # valid patient
    idx_valid = np.array(idx_valid)
    bag_data_all_withoutInstLabel = bag_data_all_withoutInstLabel[idx_valid]

    # 2.2 separate patients to train and test set
    train_list = np.array(os.listdir(os.path.join(pred_dir, 'train')))
    test_list = np.array(os.listdir(os.path.join(pred_dir, 'test')))

    train_list_tmpForIdx = np.array([path.split('_')[0] for path in train_list])
    test_list_tmpForIdx = np.array([path.split('_')[0] for path in test_list])

    data_train = []
    data_test = []
    for i in tqdm(range(bag_data_all_withoutInstLabel.shape[0]), desc='Scanning bags without GT instance label'):
        patient_i_idx = bag_data_all_withoutInstLabel[i, 0].split('/')[-1]
        patient_pathologyLabel = bag_data_all_withoutInstLabel[i, 2:6].astype(int)
        if patient_i_idx in train_list_tmpForIdx:
            patient_i_instancePred = pd.read_csv(os.path.join(pred_dir, 'train', train_list[train_list_tmpForIdx == patient_i_idx][0])).to_numpy()
            top_three_pred = np.argsort(patient_i_instancePred[:, 1])[-3:]
            data_train.append([patient_i_idx, patient_i_instancePred[top_three_pred, 0:2], patient_pathologyLabel])
        elif patient_i_idx in test_list_tmpForIdx:
            patient_i_instancePred = pd.read_csv(os.path.join(pred_dir, 'test', test_list[test_list_tmpForIdx == patient_i_idx][0])).to_numpy()
            top_three_pred = np.argsort(patient_i_instancePred[:, 1])[-3:]
            data_test.append([patient_i_idx, patient_i_instancePred[top_three_pred, 0:2], patient_pathologyLabel])
        else:
            pass
    return data_train, data_test, data_independent_test


def gather_external_Pathology(root_dir=''):
    all_imgs_along_patient = glob(os.path.join(root_dir, '*/*/'))

    # 1.2 index ground truth instances for patients with instance label
    data_independent_test = []
    for i in tqdm(range(len(all_imgs_along_patient)), ascii=True, desc='Scanning external datas with GT instance label'):
        patient_path = all_imgs_along_patient[i][:-3]

        for pathology_type, pathology_type_code in zip(['1', '3', '4', '5'],
                                                       [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]):
            pos_patches = glob(os.path.join(patient_path, pathology_type, '*.jpg'))
            if len(pos_patches) == 0:
                pass
            else:
                for pos_patches_i in pos_patches:
                    data_independent_test.append([
                        patient_path.split('/')[-1],
                        np.concatenate([np.array([pos_patches_i]).astype(object)[:, None], np.ones(1, dtype=float)[:, None]], axis=1),
                        pathology_type_code
                    ])

    return data_independent_test


def split_GT_pathology_alongPatient(ds, split=0.7):
    # this function should receive data_independent_test processed by gather_align_EndoImg_Center_forPathology
    # this function split ds into train and test along patients
    ds_tmp_forIdx = np.array(ds, dtype=object)
    patient_id_all = np.unique(ds_tmp_forIdx[:, 0])
    N = len(patient_id_all)
    patient_id_all = patient_id_all[np.random.permutation(N)]

    patient_id_train = patient_id_all[:int(split * N)]
    patient_id_test = patient_id_all[int(split * N):]
    ds_train = []
    ds_test = []
    for data_i in ds:
        if data_i[0] in patient_id_train:
            ds_train.append(data_i)
        elif data_i[0] in patient_id_test:
            ds_test.append(data_i)
        else:
            raise
    return ds_train, ds_test


class Endo_img_MIL_all_center_withPathologyLabel(torch.utils.data.Dataset):
    # @profile
    def __init__(self, ds, transform=None, num_instances_per_bag=3, task_classes=2, preload=True, certainty_threshold=0.75):
        self.ds = deepcopy(ds)
        self.transform = transform
        self.num_instances_per_bag = num_instances_per_bag
        self.task_classes = task_classes
        self.preload = preload
        self.certainty_threshold = certainty_threshold
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47403994, 0.29777905, 0.18895507],
                                     std=[0.2768913, 0.2088029, 0.16106644])
            ])

        # # preprocess
        # for i in tqdm(range(len(self.ds)), desc='Preprocessing: fix each patient with same number of instances', ascii=True):
        #     num_instances = len(self.ds[i][1])
        #     if num_instances > self.num_instances_per_bag:
        #         # new_idx = np.random.choice(self.ds[i][1].shape[0], size=self.num_instances_per_bag, replace=False)
        #         # self.ds[i][1] = self.ds[i][1][new_idx]
        #         self.ds[i][1] = self.ds[i][1][-self.num_instances_per_bag:]
        #     elif num_instances < self.num_instances_per_bag:
        #         repeat_idx = np.random.choice(self.ds[i][1].shape[0], size=int(self.num_instances_per_bag - num_instances), replace=True)
        #         new_idx = np.concatenate([np.arange(self.ds[i][1].shape[0]), repeat_idx])
        #         self.ds[i][1] = self.ds[i][1][new_idx]

        # # filter patient with low certainty
        # assert self.num_instances_per_bag == 1
        # self.ds_instance_certainty = np.array([self.ds[i][1] for i in range(len(self.ds))]).squeeze()[:, 1].astype(float)
        # self.ds_new = []
        # for i in tqdm(range(len(self.ds)), desc='Filtering: filter patients with low certainty', ascii=True):
        #     if self.ds_instance_certainty[i] < self.certainty_threshold:
        #         continue
        #     else:
        #         self.ds_new.append(self.ds[i])
        # self.ds = self.ds_new

        self.ds_labelPart = np.array([self.ds[i][2] for i in range(len(self.ds))])

        if self.task_classes == 4:
            self.ds_labelPart_merged = self.ds_labelPart.argmax(axis=1)
        elif self.task_classes == 3:
            self.ds_labelPart_merged = np.zeros([self.ds_labelPart.shape[0], 3], dtype=int)
            self.ds_labelPart_merged[:, 0] = self.ds_labelPart[:, 0]
            self.ds_labelPart_merged[:, 1] = self.ds_labelPart[:, 1] | self.ds_labelPart[:, 2]
            self.ds_labelPart_merged[:, 2] = self.ds_labelPart[:, 3]
            self.ds_labelPart_merged = self.ds_labelPart_merged.argmax(axis=1)
        elif self.task_classes == 2:
            self.ds_labelPart_merged = np.zeros([self.ds_labelPart.shape[0], 2], dtype=int)
            self.ds_labelPart_merged[:, 0] = self.ds_labelPart[:, 0]
            self.ds_labelPart_merged[:, 1] = self.ds_labelPart[:, 1] | self.ds_labelPart[:, 2] | self.ds_labelPart[:, 3]
            self.ds_labelPart_merged = self.ds_labelPart_merged.argmax(axis=1)

        if self.preload:
            self.data_preload = []
            for index in tqdm(range(len(self.ds)), desc='Preloading images', ascii=True):
                instance_all = []
                for i in range(self.num_instances_per_bag):
                    instance_i_path = self.ds[index][1][i, 0]
                    instance_i = io.imread(instance_i_path)
                    instance_i = self.transform(Image.fromarray(np.uint8(instance_i), 'RGB'))
                    instance_all.append(instance_i)
                instance_all = np.stack(instance_all, axis=0)
                self.data_preload.append(instance_all)
        print("")

    def __getitem__(self, index):
        if self.preload:
            instance_all = self.data_preload[index]
        else:
            instance_all = []
            for i in range(self.num_instances_per_bag):
                instance_i_path = self.ds[index][1][i, 0]
                instance_i = io.imread(instance_i_path)
                instance_i = self.transform(Image.fromarray(np.uint8(instance_i), 'RGB'))
                instance_all.append(instance_i)
            instance_all = np.stack(instance_all, axis=0)
        patch_label = self.ds_labelPart_merged[index]

        return instance_all, [patch_label], index

    def __len__(self):
        return len(self.ds)


if __name__ == '__main__':
    ds_External_test_withInstLabel = gather_external_Pathology(root_dir='')
    ds_pathology_train, ds_pathology_test = split_GT_pathology_alongPatient(ds_External_test_withInstLabel)

    ds_train, ds_test, ds_test_withInstLabel = gather_align_EndoImg_Center_forPathology(
        root_dir='',
        pred_dir='',
        labeled_dir=''
    )
    ds_pathology_train, ds_pathology_test = split_GT_pathology_alongPatient(ds_test_withInstLabel)
    loader_withInstLabel_train = Endo_img_MIL_all_center_withPathologyLabel(ds_pathology_train, num_instances_per_bag=1)
    loader_withInstLabel_test = Endo_img_MIL_all_center_withPathologyLabel(ds_pathology_test, num_instances_per_bag=1)

    loader_train = Endo_img_MIL_all_center_withPathologyLabel(ds_train, num_instances_per_bag=1)
    loader_test = Endo_img_MIL_all_center_withPathologyLabel(ds_test, num_instances_per_bag=1)
    loader_test_withInstLabel = Endo_img_MIL_all_center_withPathologyLabel(ds_test_withInstLabel, num_instances_per_bag=1)
    print(len(loader_train), loader_train.ds_labelPart_merged.sum(axis=0))
    print(len(loader_test), loader_test.ds_labelPart_merged.sum(axis=0))
    print(len(loader_test_withInstLabel), loader_test_withInstLabel.ds_labelPart.sum(axis=0))
    print("")
