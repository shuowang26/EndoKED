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
from datasets.data_processing_refining.a1_generate_all_pos_polyp_path_as_csv import load_img_and_save_pos_path
from datasets.data_processing_refining.sam_utils import *
import datasets.data_processing_refining.a2_dataset_loader_all_save_pseudolabel as polyp
from tqdm import tqdm
from reference_codes.a_segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from reference_codes.b_polyp_PVT.lib.pvt import PolypPVT 
from reference_codes.b_polyp_PVT.utils.dataloader_zhongshan import zhongshan_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--normalize', type=bool, default=True, help='testing size')
parser.add_argument('--data_name', type=str, default='cvc-db', help='testing size')
parser.add_argument('--pth_path', type=str, default='/data/PROJECTS/Endo_GPT/reference_codes/HSNet-main/best_checkpoints/31PolypPVT-best_ETIS_0.723.pth')

def _validate(model=None, data_loader=None, predictor=None,):

    print('01#############Pseudo Label Seeds from SAM Started!!!')
    model.eval()
    dice_all = []
    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()
        num1 = len(data_loader.images)

        for i in tqdm(range(num1),total=len(data_loader.images), ncols=100, ascii=" >="):
            image, image_raw,gt_, name = data_loader.load_data()

            image = image.cuda()
            P1,P2 = model(image)
            res = F.upsample(P1+P2, size=gt_.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res =(res - res.min()) / (res.max() - res.min() + 1e-8)
            seg_pred = res
            seg_pred = np.uint8(res>= 1e-3)
        
            if predictor is None:
                """####### save label from Preds #################"""
                # boxes = from_mask2box_singlebox(seg_pred)
                # if boxes is None:
                #     continue
                label = seg_pred * 255

            else:
                """####### save label from SAM #################"""
                # inputs_  = imutils.denormalize_img(inputs_ )[0].permute(1,2,0).cpu().numpy()
                # inputs_raw = imutils.denormalize_img(inputs_raw.cpu())[0].permute(1,2,0).type(torch.uint8).numpy()
                predictor.set_image(image_raw)
                boxes = from_mask2box_singlebox_eval(seg_pred)
                # boxes = from_mask2box(cam_label)
                # if boxes is None:
                #     continue
                if len(boxes.shape) > 1 and boxes.shape[0] > 1:
                    boxes = torch.tensor(boxes).cuda()
                    boxes = predictor.transform.apply_boxes_torch(boxes, image_raw.shape[:2])
                    masks_pred, scores, logits = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=boxes,
                    multimask_output=False,)
                    mask_sum = np.zeros_like(seg_pred)
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
                # label = F.interpolate(label, size=(H,W), mode='bilinear', align_corners=False)

            # plt.subplot(121)
            # plt.imshow(label)
            # plt.subplot(122)
            # plt.imshow(gt_)
            # plt.show()
            dice,_ = cal_dice(label,gt_)
            dice_all.append(dice)
        mean_dice = np.stack(dice_all,axis=0).mean()
        print(f'The dice for {args.data_name} is :{mean_dice}')
            
    return 

def validate(image_root,gt_root,args=None, predictor=None):
 
    test_loader = polyp.test_dataset(image_root, gt_root, 352, normalize=args.normalize)
    model = PolypPVT()
    new_state_dict = {}
    trained_state_dict = torch.load(args.model_path, map_location='cpu')
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    _validate(model=model, data_loader=test_loader, predictor=predictor)
    torch.cuda.empty_cache()

    print(f'01#############{args.data_name}_Pseudo Label Seeds from SAM Finished!!!')
    return True

if __name__ == "__main__":

    args = parser.parse_args()
    args.normalize = False
    data_list = ['CVC-300', 'CVC-ClinicDB-Selected', 'kvasir-Selected_0.909', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    model_path = ['/data/PROJECTS/Endo_GPT/reference_codes/Polyp-PVT-main/00_sota_savepoints/CVC-300_0.916_e23/11PolypPVT.pth',\
            '/data/PROJECTS/Endo_GPT/reference_codes/Polyp-PVT-main/00_sota_savepoints/CVC-DB0.919/20PolypPVT.pth',\
            '/data/PROJECTS/Endo_GPT/reference_codes/Polyp-PVT-main/00_sota_savepoints/kvasir_0.911/1PolypPVT.pth',\
            '/data/PROJECTS/Endo_GPT/reference_codes/Polyp-PVT-main/00_sota_savepoints/CVC-Colon_0.790/28PolypPVT.pth',\
            '/data/PROJECTS/Endo_GPT/reference_codes/Polyp-PVT-main/00_sota_savepoints/ETIS-LaribPolypDB_0.817/model_Preds_pth/87PolypPVT.pth'
                ]

    for _data_name,model_path in zip(data_list,model_path):
        args.data_name = _data_name
        args.model_path = model_path
        image_root = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/dataset/TestDataset/{}/images/'.format(_data_name)
        gt_root = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/dataset/TestDataset/{}/masks/'.format(_data_name)
        """############Load SAM#############"""
        #### SET UP 
        sam_checkpoint = "/data/PROJECTS/Endo_GPT/05_segment-anything-main/pretrained/sam_vit_h_4b8939.pth"
        device = "cuda"
        model_type = "default"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        # predictor = SamPredictor(sam)
        predictor = None 

        validate(image_root,gt_root,args=args,predictor=predictor)
