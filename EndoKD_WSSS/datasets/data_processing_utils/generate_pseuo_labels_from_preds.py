import argparse
import os
import sys

sys.path.append(".")

from collections import OrderedDict
import imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import dataset_loader_zhongshan_save_pseudolabel as voc
from datasets.dataset_loader_zhongshan_save_pseudolabel import load_img_label_info_from_csv, load_test_img_mask
from model.model_seg_neg import network
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs

parser = argparse.ArgumentParser()
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
# parser.add_argument("--model_path", default="workdir_voc_final2/2022-11-04-01-50-48-441426/checkpoints/model_iter_20000.pth", type=str, help="model_path")
# parser.add_argument("--model_path", default="/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/work_dir_voc_wseg/2023-04-20-16-45-19-197102/checkpoints/model_iter_6000.pth", type=str, help="model_path")
parser.add_argument("--model_path", default="/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/work_dir_voc_wseg/2023-06-13-00-55-45-128778/checkpoints/model_iter_20000.pth", type=str, help="model_path")

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--data_folder", default='../VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=2, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--scales", default=(1.0, 1.5, 1.25), help="multi_scales for seg")
parser.add_argument("--num_workers", default=8, type=int, help="num_workers")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")


#####


parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", default=True, help="fix random seed")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
def _validate(model=None, data_loader=None, args=None):

    model.eval()
    color_map = plt.get_cmap("Blues")

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()

        gts, seg_pred = [], []
        name_lst = []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            path_name, inputs, cls_label = data

            name = 'ZS00' + path_name[0].split('ZS00')[-1].replace('/','_')
            inputs = inputs.cuda()
            cls_label = cls_label.cuda()

            _, _, h, w = inputs.shape
            seg_list = []
            for sc in args.scales:
                _h, _w = int(h*sc), int(w*sc)

                _inputs  = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs = model(inputs_cat,)[1].detach().cpu()
                segs = F.interpolate(segs, size=inputs.shape[2:], mode='bilinear', align_corners=False)

                # seg = torch.max(segs[:1,...], segs[1:,...].flip(-1))
                seg = segs[:1,...] + segs[1:,...].flip(-1)

                seg_list.append(seg)
            seg = torch.max(torch.stack(seg_list, dim=0), dim=0)[0]
            
            seg_pred += list(torch.argmax(seg, dim=1).cpu().numpy().astype(np.int16))

            np.save(args.logits_dir + "/" + name + '.npy', {"msc_seg":seg.cpu().numpy()})
            name_lst.append(name)
        with open('/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/polyp_zhongshan_pos/selected_polyp_img_name.txt','a') as f:
            for item in name_lst:
                f.write(f'{item}\n')
 
    
    return name_lst


def crf_proc(name_lst):
    print("crf post-processing...")

    name_list = name_lst
    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=1,   # 3
        pos_w=1,        # 3
        bi_xy_std=121,  # 121, 140
        bi_rgb_std=5,   # 5, 5
        bi_w=4,         # 4, 5
    )

    def _job(i):

        name = name_list[i]
        name_ = name_list[i].replace('.jpg','')

        logit_name = args.logits_dir + "/" + name + ".npy"

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_seg']

        image_name = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/中山/图像/' + name_list[i].replace('_','/')
        image = imageio.imread(image_name).astype(np.float32)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()
        # prob = logit[0]

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        #print(pred.shape)
        imageio.imsave(args.segs_dir + "/" + name_ + ".png", np.squeeze(pred).astype(np.uint8))
        imageio.imsave(args.segs_rgb_dir + "/" + name_ + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred
    
    n_jobs = int(os.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])
    return


def validate(args=None):

    with open ('/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/polyp_zhongshan_pos/selected_polyp_path.txt','r') as f:

        img_path_list_train = f.readlines()
    label_train = np.ones((len(img_path_list_train),1))

    
    train_ds =  voc.Endo_img_WSSS(img_path_list_train,label_train,aug=False,)
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


    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    # new_state_dict.pop("conv.weight")
    # new_state_dict.pop("aux_conv.weight")

    model.load_state_dict(state_dict=new_state_dict, strict=False)
    model.eval()

    name_lst = _validate(model=model, data_loader=train_loader, args=args)
    torch.cuda.empty_cache()

    crf_proc(name_lst)
    
    return True

if __name__ == "__main__":

    args = parser.parse_args()

    base_dir = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/visual_logits_generate_pseudolabel_zhongshan'
    args.logits_dir = os.path.join(base_dir, "segs/logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, "segs/seg_preds", args.infer_set)
    args.segs_rgb_dir = os.path.join(base_dir, "segs/seg_preds_rgb", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.segs_rgb_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)

    print(args)
    validate(args=args)