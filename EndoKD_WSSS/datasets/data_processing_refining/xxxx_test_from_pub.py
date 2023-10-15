import argparse
import os
import sys

sys.path.append("/data/PROJECTS/Endo_GPT/EndoGPT_WSSS")
from collections import OrderedDict

import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets.a_zhongshan_2wpublic_all.a1_generate_all_pos_polyp_path_as_csv import load_public_train_path
from datasets.a_zhongshan_2wpublic_all.sam_utils import *
import datasets.a_zhongshan_2wpublic_all.a2_dataset_loader_all_save_pseudolabel as voc
from model.model_seg_neg import network
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.camutils import cam_to_label, get_valid_cam, multi_scale_cam2
from utils.pyutils import AverageMeter, format_tabs
from a_segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

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


def _validate(model=None, data_loader=None, args=None, predictor=None,filtered_csv_path=None):

    print('01#############Pseudo Label Seeds from SAM Started!!!')
    model.eval()
    cam_dir = args.cam_dir

    os.makedirs(cam_dir, exist_ok=True)

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()
        with open (filtered_csv_path,'w') as f:
            writer = csv.writer(f)
            header = ['filtered_img_path','filtered_label_name']
            writer.writerow(header)
            for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

                name, inputs_, cls_label, max, data_info = data
                inputs_ = inputs_.cuda()
                # img = imutils.denormalize_img(inputs_)[0].permute(1,2,0).cpu().numpy()
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
                
                if predictor is None:
                    """####### save label from CAMAM #################"""
                    label = cam_label * 255
                    cv2.imwrite(os.path.join(cam_dir, name[0]), label)

                else:
                    """####### save label from SAM #################"""
                    # inputs_  = imutils.denormalize_img(inputs_ )[0].permute(1,2,0).cpu().numpy()
                    inputs_ = (inputs_[0].cpu() * max).permute(1,2,0).type(torch.uint8).numpy()
                    predictor.set_image(inputs_)
                    boxes = from_mask2box_singlebox(cam_label)
                    # boxes = from_mask2box(cam_label)

                    if boxes is None:
                        continue
                    if len(boxes.shape) > 1 and boxes.shape[0] > 1:
                        boxes = torch.tensor(boxes).cuda()
                        boxes = predictor.transform.apply_boxes_torch(boxes, inputs_.shape[:2])
                        masks_pred, scores, logits = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=boxes,
                        multimask_output=False,)
                        mask_sum = np.zeros_like(cam_label)
                        mask_sum = masks_pred.sum(dim=0).cpu().numpy()
                        final_mask = np.uint8(mask_sum != 0)[0]

                    else:
                        masks_pred, scores, logits = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=boxes,
                        multimask_output=False,
                    )
                        final_mask = masks_pred[0]

                    label = final_mask * 255
                    cv2.imwrite(os.path.join(cam_dir, name[0]), label)
                    writer.writerow([data_info[0][0],data_info[1][0]])
    return 

def validate(img_path_list_train,label_name_lst,args=None,filtered_csv_path=None):

    """############Load SAM#############"""
    ##### SET UP 
    sam_checkpoint = "/data/PROJECTS/Endo_GPT/05_segment-anything-main/pretrained/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)
    
 
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

    _validate(model=model, data_loader=train_loader, args=args,predictor=predictor,filtered_csv_path=filtered_csv_path)
    torch.cuda.empty_cache()

    print('01#############Pseudo Label Seeds from SAM Finished!!!')
    return True

if __name__ == "__main__":

    args = parser.parse_args()
    args.model_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/model_iter_8000.pth'

    args.cam_dir = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/' + '0_CAM_seeds'
    os.makedirs(args.cam_dir, exist_ok=True)
    root_dir_train = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/dataset/TrainDataset/images/'
    root_dir_test = "/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/CVC-ClinicDB_datasets/CVC-ClinicDB_datasets/CVC-ClinicDB_PNG_datasets/"
    public_2w_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/public_2w/images/'
    csv_save_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/a_zhongshan_2wpublic_all/pub_train.csv'
    filtered_csv_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/a_zhongshan_2wpublic_all/filtered_pub_train.csv'
    img_path_list_train,label_name_lst = load_public_train_path(root_dir_train,csv_save_path)

    validate(img_path_list_train,label_name_lst,args=args,filtered_csv_path=filtered_csv_path)


