import argparse
import os
import sys

sys.path.append("/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/")

from collections import OrderedDict
import imageio.v2 as imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import datasets.hello as voc
from datasets.hello import load_test_img_mask
from model.model_seg_neg import network
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
# parser.add_argument("--model_path", default="workdir_voc_final2/2022-11-04-01-50-48-441426/checkpoints/model_iter_20000.pth", type=str, help="model_path")
# parser.add_argument("--model_path", default="/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/work_dir_voc_wseg/2023-04-20-16-45-19-197102/checkpoints/model_iter_6000.pth", type=str, help="model_path")
parser.add_argument("--model_path", default="/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/pretrained/checkpoints/preds_sota_0.62miou.pth", type=str, help="model_path")

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--data_folder", default='../VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=2, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--scales", default=[1.0], help="multi_scales for seg")
parser.add_argument("--num_workers", default=8, type=int, help="num_workers")

def _validate(model=None, data_loader=None, args=None):

    model.eval()
    color_map = plt.get_cmap("Blues")

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()

        gts, seg_pred = [], []
        name_lst = []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            inputs  = F.interpolate(inputs, size=[448, 448], mode='bilinear', align_corners=False)


            _, _, h, w = inputs.shape
            seg_list = []
            for sc in args.scales:
                _h, _w = int(h*sc), int(w*sc)

                inputs  = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                # inputs_ = torch.cat([inputs, inputs.flip(-1)], dim=0)

                segs = model(inputs,)[1]
                segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

                # seg = torch.max(segs[:1,...], segs[1:,...].flip(-1))
                # seg = segs[:1,...] + segs[1:,...].flip(-1)

                seg_list.append(segs)
            seg = torch.max(torch.stack(seg_list, dim=0), dim=0)[0]
            
            seg_pred += list(torch.argmax(seg, dim=1).cpu().numpy().astype(np.int16))

            pred = torch.argmax(seg, dim=1).cpu().numpy().astype(np.int16)
            imageio.imsave(args.segs_dir + "/" + name[0] + ".png", np.squeeze(pred).astype(np.uint8))
            imageio.imsave(args.segs_rgb_dir + "/" + name[0] + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
            gts += list(labels.cpu().numpy().astype(np.int16))

            np.save(args.logits_dir + "/" + name[0] + '.npy', {"msc_seg":seg.cpu().numpy()})
            name_lst.append(name[0])
    seg_score = evaluate.scores(gts, seg_pred,num_classes=2)

    print(format_tabs([seg_score], [ "seg_pred"], cat_list=('Bkg','endo')))
    
    return seg_score,name_lst

def validate(args=None):

    root_dir_train = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/network_MIL_prediction_results/train/'
    root_dir_test = "/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/dataset/TestDataset/test/"
    # root_dir_test = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/dataset/TestDataset/CVC-ClinicDB-all/'
    # root_dir_test = '/data/PROJECTS/Endo_GPT/Datasets/test'
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
        aux_layer = -2
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    # new_state_dict.pop("conv.weight")
    # new_state_dict.pop("aux_conv.weight")

    model.load_state_dict(state_dict=new_state_dict, strict=False)
    model.eval()

    seg_score,name_lst = _validate(model=model, data_loader=val_loader, args=args)
    torch.cuda.empty_cache()

    # crf_score = crf_proc(name_lst)
    
    return True

if __name__ == "__main__":

    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints")[0]
    args.logits_dir = os.path.join(base_dir, "segs_0.62miou/logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, "segs_0.62miou/seg_preds", args.infer_set)
    args.segs_rgb_dir = os.path.join(base_dir, "segs_0.62miou/seg_preds_rgb", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.segs_rgb_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)

    print(args)
    validate(args=args)