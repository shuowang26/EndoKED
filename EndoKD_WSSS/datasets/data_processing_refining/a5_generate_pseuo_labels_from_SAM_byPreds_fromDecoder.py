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
from tqdm import tqdm
from reference_codes.a_segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from reference_codes.b_polyp_PVT.lib.pvt import PolypPVT 
from reference_codes.b_polyp_PVT.utils.dataloader_zhongshan import zhongshan_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--normalize', type=bool, default=True, help='testing size')
parser.add_argument('--pth_path', type=str, default='/data/PROJECTS/Endo_GPT/reference_codes/HSNet-main/best_checkpoints/31PolypPVT-best_ETIS_0.723.pth')

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
            num1 = len(data_loader.images)
            for i in tqdm(range(num1),total=len(data_loader.images), ncols=100, ascii=" >="):
                image, image_,name, data_info = data_loader.load_data()
                image = image.cuda()
                P1,P2 = model(image)
                res = F.upsample(P1+P2, size=image_.shape[:2], mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res =(res - res.min()) / (res.max() - res.min() + 1e-8)
                seg_pred = res
                seg_pred = np.uint8(res>= 0.01)
            
                if predictor is None:
                    """####### save label from Preds #################"""
                    boxes = from_mask2box_singlebox(seg_pred)
                    if boxes is None:
                        continue
                    label = seg_pred * 255
                    cv2.imwrite(os.path.join(pred_dir, name), label)
                    writer.writerow([data_info[0],data_info[1]])

                else:
                    """####### save label from SAM #################"""
                    # inputs_  = imutils.denormalize_img(inputs_ )[0].permute(1,2,0).cpu().numpy()
                    # inputs_raw = imutils.denormalize_img(inputs_raw.cpu())[0].permute(1,2,0).type(torch.uint8).numpy()
                    predictor.set_image(image_)
                    boxes = from_mask2box_singlebox(seg_pred)
                    # boxes = from_mask2box(cam_label)
                    if boxes is None:
                        continue
                    if len(boxes.shape) > 1 and boxes.shape[0] > 1:
                        boxes = torch.tensor(boxes).cuda()
                        boxes = predictor.transform.apply_boxes_torch(boxes, image_.shape[:2])
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

                    cv2.imwrite(os.path.join(pred_dir, name), label)
                    writer.writerow([data_info[0],data_info[1]])
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
 
    train_loader = zhongshan_dataset(img_path_list_train,label_name_lst, 352,normalize=args.normalize)
    model = PolypPVT()
    new_state_dict = {}
    trained_state_dict = torch.load(args.model_path, map_location='cpu')
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    _validate(model=model, data_loader=train_loader, args=args,predictor=predictor,filtered_csv_path=filtered_csv_path)
    torch.cuda.empty_cache()

    print('01#############Pseudo Label Seeds from SAM Finished!!!')
    return True

if __name__ == "__main__":

    args = parser.parse_args()
    args.normalize = False
    args.model_path = '/data/PROJECTS/Endo_GPT/reference_codes/Polyp-PVT-main/00_sota_savepoints/sota_all_0.826_iter6_fromPred/53PolypPVT.pth'
    args.pred_dir = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/' + '0.826_0706_Preds_seeds_decoder2/'
    os.makedirs(args.pred_dir, exist_ok=True)
    root_dir_train = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/network_MIL_prediction_results/train/'
    public_2w_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/public_2w/V1/'
    csv_save_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/a_zhongshan_2wpublic_all/zhongshan_2wpublic_imgPath.csv'
    filtered_csv_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/a_zhongshan_2wpublic_all/0.826_0706_zfiltered_decoder_after_Preds_zhongshan_2wpublic_imgPath2.csv'
    img_path_list_train,label_name_lst = load_img_and_save_pos_path(root_dir_train,public_2w_path,csv_save_path)
    
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

    validate(img_path_list_train,label_name_lst,args=args,filtered_csv_path=filtered_csv_path, predictor=predictor)
