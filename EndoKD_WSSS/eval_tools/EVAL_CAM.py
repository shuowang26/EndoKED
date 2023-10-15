import argparse
import os
import sys

sys.path.append("/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/")
from collections import OrderedDict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets.hello import load_test_img_mask
from datasets import hello as voc
# from datasets import voc
from model.model_seg_neg import network
# from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.camutils import cam_to_label, get_valid_cam, multi_scale_cam2
from utils.pyutils import AverageMeter, format_tabs

parser = argparse.ArgumentParser()
parser.add_argument("--bkg_thre", default=0.65, type=float, help="work_dir")

# parser.add_argument("--model_path", default="/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/work_dir_voc_wseg/2023-04-03-08-37-23-976228/checkpoints/model_iter_16000.pth", type=str, help="model_path")
parser.add_argument("--model_path", default="/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/pretrained/checkpoints/model_iter_6000.pth", type=str, help="model_path")

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

    base_dir = args.model_path.split("checkpoints")[0]
    cam_dir = os.path.join(base_dir, "cam_img", args.infer_set)
    cam_aux_dir = os.path.join(base_dir, "cam_img_aux", args.infer_set)

    os.makedirs(cam_aux_dir, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)
    color_map = plt.get_cmap("jet")

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()

        gts, cams, aux_cams = [], [], []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            # img = imutils.denormalize_img(inputs)[0].permute(1,2,0).cpu().numpy()
            img = (inputs)[0].permute(1,2,0).cpu().numpy() * 255

            inputs  = F.interpolate(inputs, size=[448, 448], mode='bilinear', align_corners=False)
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            ###
            _cams, _cams_aux = multi_scale_cam2(model, inputs, [1.0, 0.5, 0.75, 1.25,1.5,1.75])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)

            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre)
            cam_aux_label = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre)

            resized_cam = get_valid_cam(resized_cam, cls_label)
            resized_cam_aux = get_valid_cam(resized_cam_aux, cls_label)

            cam_np = torch.max(resized_cam[0], dim=0)[0].cpu().numpy()
            ## save prompt
            cam_tensor = torch.max(resized_cam[0], dim=0)[0]
            max_value = cam_tensor.max()
            prompt_pos = (cam_tensor == max_value).nonzero()[0]
            # np.save(args.logits_dir + "/" + name[0] + '.npy', {"name":name[0],"cam_np":cam_np,"resized_cam_raw":resized_cam,"prompt_pos":prompt_pos.cpu().numpy(),"img":img,"mask":labels.cpu().numpy(),"cls_label":cls_label})

            cam_aux_np = torch.max(resized_cam_aux[0], dim=0)[0].cpu().numpy()
            color_map = plt.get_cmap("jet")
            cam_rgb = color_map(cam_np)[:,:,:3] * 255
            cam_aux_rgb = color_map(cam_aux_np)[:,:,:3] * 255

            alpha = 0.4
            cam_rgb = ( alpha*cam_rgb + (1-alpha)*img ) / 255
            cam_aux_rgb = (alpha*cam_aux_rgb + (1-alpha)*cam_aux_rgb) / 255 

            # plt.imshow(cam_rgb)
            # plt.show()
            # imageio.imsave(os.path.join(cam_dir, name[0] + ".jpg"), cam_rgb)
            # imageio.imsave(os.path.join(cam_aux_dir, name[0] +".jpg"), cam_aux_rgb.astype(np.uint8))

            imageio.imsave(os.path.join(cam_dir, name[0] + ".png"), imutils.encode_cmap(cam_label[0].cpu().numpy().astype(np.uint8)))
            imageio.imsave(os.path.join(cam_aux_dir, name[0] +".png"), imutils.encode_cmap(cam_aux_label[0].cpu().numpy().astype(np.uint8)))

            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            aux_cams += list(cam_aux_label.cpu().numpy().astype(np.int16))

    cam_score = evaluate.scores(gts, cams, num_classes=2)
    cam_aux_score = evaluate.scores(gts, aux_cams, num_classes=2)
    
    return format_tabs([cam_score, cam_aux_score], ["cam", "aux_cam"], cat_list=('Bkg','endo'))


def validate(args=None):
    root_dir_train = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/network_MIL_prediction_results/train/'
    root_dir_test = "/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/dataset/TestDataset/test/"
    img_path_list_test, mask_test_path = load_test_img_mask(root_dir_test)
    # img_path_list_train, label_train = load_img_label_info_from_csv(root_dir_train)

    val_dataset = voc.Endo_img_WSSS(
        img_path_list_test,
        mask_test_path,
        type='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

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

    print(model.load_state_dict(state_dict=new_state_dict, strict=True))
    model.eval()

    results = _validate(model=model, data_loader=val_loader, args=args)
    torch.cuda.empty_cache()

    print(results)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    base_dir = args.model_path.split("checkpoints")[0]
    args.logits_dir = os.path.join(base_dir, "segs/cam_logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, "segs/cam_seg_preds", args.infer_set)
    args.segs_rgb_dir = os.path.join(base_dir, "segs/cam_seg_preds_rgb", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.segs_rgb_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)

    validate(args=args)

