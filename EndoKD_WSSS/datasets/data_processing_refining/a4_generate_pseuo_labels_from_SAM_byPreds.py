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
import datasets.data_processing_refining.a2_dataset_loader_all_save_pseudolabel as voc
from model.model_seg_neg import network
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from reference_codes.a_segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
# parser.add_argument("--model_path", default="workdir_voc_final2/2022-11-04-01-50-48-441426/checkpoints/model_iter_20000.pth", type=str, help="model_path")
# parser.add_argument("--model_path", default="/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/work_dir_voc_wseg/2023-04-20-16-45-19-197102/checkpoints/model_iter_6000.pth", type=str, help="model_path")
parser.add_argument("--model_path", default="/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/pretrained/checkpoints/model_iter_6000.pth", type=str, help="model_path")

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--data_folder", default='../VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=2, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--scales", default=[1.0], help="multi_scales for seg")
parser.add_argument("--num_workers", default=8, type=int, help="num_workers")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")


#####


parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", default=True, help="fix random seed")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")

def _validate(model=None, data_loader=None, args=None, predictor=None,filtered_csv_path=None):

    print('01#############Pseudo Label Seeds from SAM Started!!!')
    model.eval()
    pred_dir = args.pred_dir

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()
        with open (filtered_csv_path,'w') as f:
            writer = csv.writer(f)
            header = ['filtered_img_path','filtered_label_name']
            writer.writerow(header)

            for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

                name, inputs, cls_label, image_raw, data_info = data
                inputs = inputs.cuda()
                _,C,H, W, = inputs.shape
                cls_label = cls_label.cuda()
                inputs_raw = image_raw[0].numpy()
                inputs  = F.interpolate(inputs, size=[448, 448], mode='bilinear', align_corners=False)

                _, _, h, w = inputs.shape
                seg_list = []
                for sc in args.scales:
                    _h, _w = int(h*sc), int(w*sc)

                    inputs  = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                    inputs_ = torch.cat([inputs, inputs.flip(-1)], dim=0)

                    segs = model(inputs_,)[1]
                    segs = F.interpolate(segs, size=(H,W), mode='bilinear', align_corners=False)

                    # seg = torch.max(segs[:1,...], segs[1:,...].flip(-1))
                    seg = segs[:1,...] + segs[1:,...].flip(-1)

                    seg_list.append(seg)
                seg = torch.max(torch.stack(seg_list, dim=0), dim=0)[0]
                
                seg_pred = torch.argmax(seg, dim=1).cpu().numpy().astype(np.uint8)[0]
            
                if predictor is None:
                    """####### save label from Preds #################"""
                    boxes = from_mask2box_singlebox(seg_pred)
                    if boxes is None:
                        continue
                    label = seg_pred * 255
                    cv2.imwrite(os.path.join(pred_dir, name[0]), label)
                    writer.writerow([data_info[0][0],data_info[1][0]])

                else:
                    """####### save label from SAM #################"""
                    # inputs_  = imutils.denormalize_img(inputs_ )[0].permute(1,2,0).cpu().numpy()
                    # inputs_raw = imutils.denormalize_img(inputs_raw.cpu())[0].permute(1,2,0).type(torch.uint8).numpy()
                    predictor.set_image(inputs_raw)
                    boxes = from_mask2box_singlebox(seg_pred)
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

                    cv2.imwrite(os.path.join(pred_dir, name[0]), label)
                    writer.writerow([data_info[0][0],data_info[1][0]])
                    # plt.subplot(141)
                    # plt.imshow(inputs_raw)
                    # plt.subplot(142)
                    # plt.imshow(seg_pred)
                    # plt.subplot(143)
                    # plt.imshow(label)
                    # plt.subplot(144)
                    # plt.imshow(inputs_.cpu()[0].permute(1,2,0).numpy())
                    # plt.show()
                    
    return 

def validate(img_path_list_train,label_name_lst,args=None,filtered_csv_path=None, predictor=None):
 
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
    args.model_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/pretrained/checkpoints/model_iter_6000_sota_0626.pth'
    args.pred_dir = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/' + '0628_Preds_seeds_strictsota'
    os.makedirs(args.pred_dir, exist_ok=True)
    root_dir_train = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/network_MIL_prediction_results/train/'
    public_2w_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/public_2w/V1/'
    csv_save_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/a_zhongshan_2wpublic_all/zhongshan_2wpublic_imgPath.csv'
    filtered_csv_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/a_zhongshan_2wpublic_all/filtered_after_Preds_zhongshan_2wpublic_imgPath_sota_strict.csv'
    img_path_list_train,label_name_lst = load_img_and_save_pos_path(root_dir_train,public_2w_path,csv_save_path)
    
    """############Load SAM#############"""
    #### SET UP 
    sam_checkpoint = "/data/PROJECTS/Endo_GPT/05_segment-anything-main/pretrained/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)
    # predictor = None 

    validate(img_path_list_train,label_name_lst,args=args,filtered_csv_path=filtered_csv_path, predictor=predictor)
