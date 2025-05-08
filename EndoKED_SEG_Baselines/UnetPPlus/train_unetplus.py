import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from unetpp import NestedUNet as UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

from utils.dataloader import get_loader
from utils.dataloader import test_dataset
from utils.metrics import Metrics
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging 
from datetime import datetime
import numpy as np
import random 
from utils.metrics import evaluate
import warnings
warnings.filterwarnings('ignore')
import logging
from utils.utils import clip_gradient, poly_lr, AvgMeter



dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

def valid(model, dataset, args):

    args.valid_data_dir = f'{args.val_root}/{dataset}/'
    image_root = '{}/images/'.format(args.valid_data_dir)
    gt_root = '{}/masks/'.format(args.valid_data_dir)
    valid_dataloader = test_dataset(image_root, gt_root, args.testsize)
    total_batch = int(len(os.listdir(gt_root)) / 1)
    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2', 'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    bar = tqdm(valid_dataloader.images)  
    for i in bar:
        image, gt, name = valid_dataloader.load_data()
        # gt /= (gt.max() + 1e-8)
        image = image.cuda()
        gt = gt.cuda()
        
        output = model(image)
        # output = pred[len(pred)-1].sigmoid()

        _recall, _specificity, _precision, _F1, _F2, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

        metrics.update(recall= _recall, specificity= _specificity, precision= _precision, F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, IoU_bg= _IoU_bg, IoU_mean= _IoU_mean)

    metrics_result = metrics.mean(total_batch)

    return metrics_result, total_batch

def train_model(
        train_loader, 
        model,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):

    total_step = len(train_loader)
    best = 0
    best_idx = 0
    n_classes = 1
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    loss_record3 = AvgMeter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader, start=1):
            images, true_masks = batch

            images = images.cuda()
            true_masks = true_masks.cuda().squeeze(1)

            with torch.autocast( 'cuda' , enabled=amp):
                masks_pred = model(images)

                if n_classes == 1:
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                else:
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )
                loss_record3.update(loss.data, args.batchsize)

                if args.local_rank == 0:
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                        '[lateral-3: {:.4f}]'.
                        format(datetime.now(), epoch, args.epochs, i, total_step,
                                loss_record3.show()))

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            # Evaluation round

        if args.local_rank == 0:
            logging.info(f"####################################Testing_EPOCH{epoch}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            # for dataset in ['kvasir-Selected_0.909','CVC-ClinicDB-Selected', 'CVC-ColonDB', 'CVC-300',  'ETIS-LaribPolypDB', 'polygen']:
        total_dice = 0.0
        total_images = 0
        # for dataset in [ 'Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen']:
        for dataset in [ 'Kvasir','CVC-ClinicDB']:

            metrics_result, num_images = valid(model, dataset,args)
            total_dice += metrics_result['F1'] * num_images
            total_images += num_images
            mdice,miou,mprecision,mrecall=metrics_result['F1'], metrics_result['IoU_poly'], metrics_result['precision'],metrics_result['recall']
            if args.local_rank == 0:
                print(f'TrainingEpoch[{epoch}]:\tTested on Dataset:[{dataset}]>>>>>>>>>')
                print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'], metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'], metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))
                logging.info(f'TrainingEpoch[{epoch}]_Tesed_on_[{dataset}]:\tDice:{mdice}\tIoU:{miou}\tPrecision:{mprecision}\tRecall:{mrecall}')  

        meandice = total_dice/total_images
        if meandice > best:
            best_idx = epoch
            best = meandice
            if args.local_rank == 0:
                torch.save(model.state_dict(), f'{args.record_path}/{args.dataset}_{epoch}_Best.pt')
                print(">>>>>>>>>>>>>>>>>Epoch %d with best mdice: %.4f" % (best_idx, best))
                logging.info(">>>>>>>>>>>>>>>>>Epoch %d with best mdice: %.4f" % (best_idx, best))

        scheduler.step(meandice)

    if args.local_rank == 0:
        torch.save(model.state_dict(), f'{args.record_path}/{args.dataset}_Last.pt')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    parser.add_argument('--epochs', type=int,
                        default=500, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=72, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--testsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_save', type=str,
                        default='C2FNet')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)

    parser.add_argument('--val_root', type=str, default='/data2/zhangruifei/polypseg')
    parser.add_argument('--dataset', type=str, default='Train_on_KvasirandDB')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')


    parser.add_argument('--tag', type=str,
                        default='scratch', help='decay rate of learning rate')
    parser.add_argument('--cpt_path', type=str,
                        default='./pretrained.pth', help='decay rate of learning rate')
    parser.add_argument('--load_pretrained',
                        default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="log tb")


    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    args = get_args()

    setup_seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', )

    if args.local_rank == 0:
        timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
        args.record_path = f'./logs/{args.dataset}/{timestamp}_{args.tag}/'
        os.makedirs(args.record_path,exist_ok=True)
        args.log_path = args.record_path + 'train_log.log'
        logging.basicConfig(filename=args.log_path,
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')


    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(input_channels=3, num_classes=args.classes,).cuda()
    if args.load_pretrained:
        print("LOADING CPT from ENDOKED")
        cpt = torch.load(args.cpt_path,map_location='cpu')
        new = {}
        for k_,v in cpt.items():
            if 'module.' in k_:
                k = k_.replace('module.','')
                new[k] = v
    
        model.load_state_dict(new)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    image_root = '{}/images/'.format(args.train_path)
    gt_root = '{}/masks/'.format(args.train_path)

    train_loader = get_loader(args, image_root, gt_root, batchsize=args.batchsize, trainsize=args.trainsize)

    train_model(
        train_loader, 
        model,
        epochs=args.epochs,
        batch_size=args.batchsize,
        learning_rate=args.lr,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp
    )
