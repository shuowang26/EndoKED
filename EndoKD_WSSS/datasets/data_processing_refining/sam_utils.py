import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from medpy import metric
import matplotlib.pyplot as plt
import cv2
import numpy as np 
from glob import glob
from tqdm import tqdm
import imageio
from PIL import Image


# calculate dice 
def calculate_metric_percase(pred, gt):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    # gt = gtnumpy()[:,0,...]
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
    
def cal_dice(outputs,label):
    # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
    # out = out.cpu().detach().numpy()
    out = outputs
    dice,hd95 = (calculate_metric_percase(out, label))

    return dice,hd95


def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
    stats = stats[stats[:,4].argsort()]
    return stats[:-1] # 排除最外层的连通图

#extract bboxes from a mask 
def from_mask2box(mask):
    box_lst = []
    bbox = mask_find_bboxs(mask)
    for b in bbox:
        x0, y0 = b[0], b[1]
        x1 = b[0] + b[2]
        y1 = b[1] + b[3]
        # print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
        # start_point, end_point = (x0, y0), (x1, y1)
        box_coor = np.array([x0, y0, x1, y1])
        box_lst.append(box_coor)
    if len(box_lst) > 0:
        return np.stack(box_lst,axis=0)
    else:
        return None

#extract biggest bbox from a mask 
def from_mask2box_singlebox(mask):
    max_area = 0.0
    bbox = mask_find_bboxs(mask)
    for b in bbox:
        x0, y0 = b[0], b[1]
        x1 = b[0] + b[2]
        y1 = b[1] + b[3]
        # print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
        # start_point, end_point = (x0, y0), (x1, y1)
        box_coor = np.array([x0, y0, x1, y1])
        max_box_coor = np.array([x0, y0, x1, y1])
        area = np.abs((x1-x0)*(y1-y0))
        if area >= max_area :
            max_area = area
            max_box_coor = np.array([x0, y0, x1, y1])
    
    ### filter 边框的一边
    H,W = mask.shape
    l_value = mask[:,0].sum() == 0
    r_value = mask[:,-1].sum() == 0
    t_value = mask[0,:].sum() == 0
    b_value = mask[-1,:].sum() == 0

    boundery_line = l_value and r_value and t_value and b_value

    ### filter 整个边框以及大的错误息肉
    large_polyp = max_area/(H * W) < 0.35 and max_area/(H * W) > 0.01

    ### filter 错误定位SAM强行产生伪标签， 即有很多错误的息肉激活
    boxes = from_mask2box(mask)
    if boxes is None:
        return None
    else:
        if len(boxes.shape) > 1 and boxes.shape[0] >= 3:
            sam_locate = False
        else:
            sam_locate = True

    if bbox.any() and large_polyp and boundery_line and sam_locate:
        return max_box_coor

    else:
        return None

    
def from_mask2box_singlebox_eval(mask):
    max_area = 0.0
    bbox = mask_find_bboxs(mask)
    for b in bbox:
        x0, y0 = b[0], b[1]
        x1 = b[0] + b[2]
        y1 = b[1] + b[3]
        # print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
        # start_point, end_point = (x0, y0), (x1, y1)
        box_coor = np.array([x0, y0, x1, y1])
        max_box_coor = np.array([x0, y0, x1, y1])
        area = np.abs((x1-x0)*(y1-y0))
        if area >= max_area :
            max_area = area
            max_box_coor = np.array([x0, y0, x1, y1])

    ### filter 错误定位SAM强行产生伪标签， 即有很多错误的息肉激活
    boxes = from_mask2box(mask)
    if boxes is None:
        return None
    else:
        return max_box_coor


# visualizarion 
def visual_box_on_img(img,box):
    start_point, end_point = (box[0],box[1]), (box[2],box[3])
    color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
    thickness = 2 # Line thickness of 1 px 
    img_box = cv2.rectangle(img, start_point, end_point, color, thickness)
    return img_box

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))    

# calculate dice 
def calculate_metric_percase(pred, gt):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    # gt = gtnumpy()[:,0,...]
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
    
def cal_dice(outputs,label):
    # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
    # out = out.cpu().detach().numpy()
    out = outputs
    dice,hd95 = (calculate_metric_percase(out, label))

    return dice,hd95

def get_dice(input,target):
    smooth = 1
    # input_flat = np.reshape(np.uint8(input>=0.01), (-1))
    input_flat = np.reshape(input, (-1))

    target_flat = np.reshape(target, (-1))
    intersection = (input_flat * target_flat)
    dice = (2 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

    return dice 

def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16),:]

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

