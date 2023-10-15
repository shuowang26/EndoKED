import argparse
import os
import sys

sys.path.append("/home/gpu_user/data/yzw/EndoGPT_WSSS")
from collections import OrderedDict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import datasets.data_processing_refining.a2_dataset_loader_all_save_pseudolabel as voc
from datasets.data_processing_refining.a1_generate_all_pos_polyp_path_as_csv import load_img_and_save_pos_path
from datasets.sam_utils import *
from model.model_seg_neg import network
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.camutils import cam_to_label, get_valid_cam, multi_scale_cam2
from utils.pyutils import AverageMeter, format_tabs

parser = argparse.ArgumentParser()
parser.add_argument("--bkg_thre", default=0.65, type=float, help="work_dir")

# parser.add_argument("--model_path", default="/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/work_dir_voc_wseg/2023-04-03-08-37-23-976228/checkpoints/model_iter_16000.pth", type=str, help="model_path")
parser.add_argument("--model_path", default="/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/work_dir_voc_wseg/SOTA_0609_0.64dice/checkpoints/model_iter_10000.pth", type=str, help="model_path")

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--data_folder", default='../VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=2, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--num_workers", default=8, type=int, help="num_workers")


def _validate(model=None, data_loader=None, args=None):

    model.eval()

    cam_dir = args.cam_dir

    os.makedirs(cam_dir, exist_ok=True)
    color_map = plt.get_cmap("jet")

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs_, cls_label = data
            inputs_ = inputs_.cuda()
            img = imutils.denormalize_img(inputs_)[0].permute(1,2,0).cpu().numpy()

            inputs  = F.interpolate(inputs_, size=[448, 448], mode='bilinear', align_corners=False)
            cls_label = cls_label.cuda()
            labels = torch.zeros_like(inputs)
            ###
            _cams, _cams_aux = multi_scale_cam2(model, inputs, [1.0, 0.5, 0.75, 1.25,1.5,1.75])
            resized_cam = F.interpolate(_cams, size=inputs_.shape[2:], mode='bilinear', align_corners=False)
            resized_cam_aux = F.interpolate(_cams_aux, size=inputs_.shape[2:], mode='bilinear', align_corners=False)

            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre)
            cam_aux_label = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre)
            
            cam_label = cam_label[0].cpu().numpy().astype(np.uint8)
            cam_aux_label = cam_aux_label[0].cpu().numpy().astype(np.uint8)
            
            label = cam_label * 255
            imageio.imsave(os.path.join(cam_dir, name[0]),label)

            """####### save label from SAM #################"""

    return 

def validate(img_path_list_train,label_name_lst,args=None):
 
    train_ds =  voc.Endo_img_WSSS(img_path_list_train,label_name_lst)
    train_loader = DataLoader(train_ds,batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False,
                              drop_last=True,)

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        aux_layer = -2
    )

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        # pooling=args.pooling,
        aux_layer=-2,
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v

    print(model.load_state_dict(state_dict=new_state_dict, strict=False))
    model.eval()

    _validate(model=model, data_loader=train_loader, args=args)
    torch.cuda.empty_cache()

    print('00#############Pseudo Label Seeds from CAM Generated!!!')
    return True

if __name__ == "__main__":

    args = parser.parse_args()
    args.model_path = '/home/gpu_user/data/yzw/EndoGPT_WSSS/work_dir_voc_wseg/2023-06-18-14-54-42-353444/checkpoints/model_iter_8000.pth'

    args.cam_dir = '/home/gpu_user/data/EndoGPT/database/Pseudo_labels_from_SAM/' + '0_CAM_seeds'
    os.makedirs(args.cam_dir, exist_ok=True)
    root_dir_train = '/home/gpu_user/data/EndoGPT/database/network_MIL_prediction_results/train/'
    root_dir_test = "/home/gpu_user/data/EndoGPT/database/息肉公开数据集/CVC-ClinicDB_datasets/CVC-ClinicDB_datasets/CVC-ClinicDB_PNG_datasets/"
    public_2w_path = '/home/gpu_user/data/EndoGPT/database/息肉公开数据集/public_2w/images/'
    csv_save_path = '/home/gpu_user/data/yzw/EndoGPT_WSSS/datasets/001_zhongshan_2wpublic_all/zhongshan_2wpublic_imgPath.csv'
    img_path_list_train,label_name_lst = load_img_and_save_pos_path(root_dir_train,public_2w_path,csv_save_path)
    validate(img_path_list_train,label_name_lst,args=args)

