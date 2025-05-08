import os
import numpy as np
import argparse
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss


import matplotlib.pyplot as plt
import random 

from models.pidnet import PIDNet
from datasets.dataloader import get_loader, test_dataset
from utils.utils_polyp import clip_gradient, adjust_lr, AvgMeter

from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
import torch.distributed as dist
from glob import glob
from torch.nn.parallel import DistributedDataParallel
import warnings
warnings.filterwarnings('ignore')


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, opt.img_size)
    dice_ls = []
    IoU = []
    precision = []
    recall = []
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res1, res, res3, = model(image) # forward
        
        
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False) # additive aggregation and upsampling
        res = res.sigmoid().data.cpu().numpy().squeeze() # apply sigmoid aggregation for binary segmentation
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        # eval Dice
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1)) > 0.5
        target_flat = np.reshape(target, (-1)) > 0.5
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

        dice_ls.append(f1_score(target_flat,input_flat))
        IoU.append(jaccard_score(target_flat,input_flat))
        precision.append(precision_score(target_flat,input_flat))
        recall.append(recall_score(target_flat,input_flat))

    return  (np.mean(dice_ls),
            np.mean(IoU),
            np.mean(precision),
            np.mean(recall),num1)

def train(train_loader, model, optimizer, epoch, test_path, model_name = 'PIDNET-S'):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25] 
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            # ---- forward ----
            outputs = model(images) # x_extra_p, x_, x_extra_d

            h, w = gts.size(2), gts.size(3)
            ph, pw = outputs[0].size(2), outputs[0].size(3)
            if ph != h or pw != w:
                for i in range(len(outputs)):
                    outputs[i] = F.interpolate(outputs[i], size=(
                        h, w), mode='bilinear', align_corners=True)

            
            # ---- loss function ----
            sem_loss = CrossEntropy()
            sb_loss = nn.BCEWithLogitsLoss()
                                    
            bd_loss = BondaryLoss()

            loss_s = sem_loss(outputs[:-1], gts)
            loss_b = bd_loss(outputs[-1], gts)
            loss_sb = sb_loss(outputs[-2], gts)

            loss = loss_s + loss_b + loss_sb
        
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        # ---- train visualization ----
        if opt.local_rank == 0:
        
            if i % 20 == 0 or i == total_step:
                logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    ' loss: {:0.4f}]'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                            loss_record.show()))
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    ' loss: {:0.4f}]'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                            loss_record.show()))
    # save model 
    if opt.local_rank == 0:
        save_path = opt.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    # torch.save(model.state_dict(), save_path + '' + model_name + '_last.pth')
    # choose the best model

    global dict_plot
   
    if (epoch + 1) % 1 == 0:
        total_dice = 0
        total_images = 0
        # for dataset in ['kvasir-Selected_0.909','CVC-ClinicDB-Selected', 'CVC-ColonDB', 'CVC-300',  'ETIS-LaribPolypDB', 'polygen']:
        # for dataset in [ 'Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen_all']:
        for dataset in ['CVC-ClinicDB', 'Kvasir']:
    
            dataset_dice, miou, mprecision, mrecall, n_images = test(model, test_path, dataset)
            total_dice += (n_images*dataset_dice)
            total_images += n_images
            if opt.local_rank == 0:
                logging.info('epoch: {}, dataset: {}, dice: {}, miou: {}, precision: {}, recall: {}'.format(epoch, dataset, dataset_dice,miou,mprecision,mrecall))
                print('epoch: {}, dataset: {}, dice: {}, miou: {}, precision: {}, recall: {}'.format(epoch, dataset, dataset_dice,miou,mprecision,mrecall))
            # dict_plot[dataset].append(dataset_dice)
        meandice = total_dice/total_images
        dict_plot['test'].append(meandice)
        if opt.local_rank == 0:
            print('Validation dice score: {}'.format(meandice))
            logging.info('Validation dice score: {}'.format(meandice))
            if meandice > best:
                # print('##################### Dice score improved from {} to {}'.format(best, meandice))
                logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
                best = meandice
                torch.save(model.state_dict(), save_path + '' + model_name + f'{epoch}_best.pth')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'PolypPVT-CASCADE'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        default='Train_on_KvasirandDB', help='Train_on_ZhongshanandKvasirandDB')

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=512, help='training batch size')

    parser.add_argument('--img_size', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=200, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='/data/Datasets/息肉数据集/息肉公开数据集/dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='/data/Datasets/息肉数据集/息肉公开数据集/dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    parser.add_argument('--pt_path', type=str,
                        default='pvt_v2_b2.pth')

    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
    parser.add_argument("--gpu_number", default=1, type=int, help="local_rank")
    parser.add_argument('--seed', type=int,
                        default=0, help='gradient clipping margin')

    parser.add_argument('--tag', type=str,
                        default='scratch', help='decay rate of learning rate')
    parser.add_argument('--cpt_path', type=str,
                        default='./pretrained.pth', help='decay rate of learning rate')
    parser.add_argument('--load_pretrained',
                        default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="log tb")
    parser.add_argument('--imgnet_pretrained',
                        default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="log tb")

    opt = parser.parse_args()

    # ---- build models ----
    #torch.cuda.set_device(2)  # set your gpu device

    setup_seed(opt.seed)
    torch.cuda.set_device(opt.local_rank)
    dist.init_process_group(backend='nccl', )

    if opt.local_rank == 0:
        timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
        opt.record_path = f'./logs/{opt.dataset}/{timestamp}_{opt.tag}/'
        opt.save_path = opt.record_path
        os.makedirs(opt.record_path,exist_ok=True)
        opt.log_path = opt.record_path + 'train_log.log'
        logging.basicConfig(filename=opt.log_path,
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = PIDNet(m=2, n=3, num_classes=1, planes=32, ppm_planes=96, head_planes=128, augment=True)
    model.cuda()

    if opt.imgnet_pretrained:
        pretrained_state = torch.load('./PIDNet_S_ImageNet.pth', map_location='cpu')['state_dict'] 
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict = False)

    elif opt.load_pretrained:
        print("LOADING CPT from ENDOKED")
        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>LOADING CPT from ENDOKED")
        cpt = torch.load(opt.cpt_path,map_location='cpu')
        new = {}
        for k_,v in cpt.items():
            if 'module.' in k_:
                k = k_.replace('module.','')
                new[k] = v
        model.load_state_dict(new)

    model = DistributedDataParallel(model, device_ids=[opt.local_rank], find_unused_parameters=True)

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    if opt.local_rank == 0:
        logging.info(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    items = sorted(glob(f'{image_root}/*'))
    gt_items = sorted(glob(f'{gt_root}/*'))
        
    train_loader = get_loader(opt,image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    if opt.local_rank == 0:
        logging.info("Start Training>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Start Training>>>>>>>>>>>>>>>>>>>>>>>>")

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.test_path, model_name = model_name)
