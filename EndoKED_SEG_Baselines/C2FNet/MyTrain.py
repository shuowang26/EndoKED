import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.C2FNet import C2FNet
from utils.dataloader import get_loader
from utils.dataloader import test_dataset
from utils.utils import clip_gradient, poly_lr, AvgMeter
import torch.nn.functional as F
from utils.AdaX import AdaXW
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


def valid(model, dataset, opt):

    opt.valid_data_dir = f'{opt.val_root}/{dataset}/'
    image_root = '{}/images/'.format(opt.valid_data_dir)
    gt_root = '{}/masks/'.format(opt.valid_data_dir)
    valid_dataloader = test_dataset(image_root, gt_root, opt.testsize)
    total_batch = int(len(os.listdir(gt_root)) / 1)
    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2', 'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    bar = tqdm(valid_dataloader.images)  
    for i in bar:
        image, gt, name = valid_dataloader.load_data()
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        gt = gt.cuda()
        
        pred = model(image)
        output = pred[len(pred)-1].sigmoid()

        _recall, _specificity, _precision, _F1, _F2, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

        metrics.update(recall= _recall, specificity= _specificity, precision= _precision, F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, IoU_bg= _IoU_bg, IoU_mean= _IoU_mean)

    metrics_result = metrics.mean(total_batch)

    return metrics_result, total_batch


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer):
    model.train()
    best = 0
    best_idx = 0
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record3 = AvgMeter()
    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(opt.trainsize*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                lateral_map_3 = model(images)
                # ---- loss function ----
                loss3 = structure_loss(lateral_map_3, gts)
                loss = loss3
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record3.update(loss3.data, opt.batchsize)
            # ---- train visualization ----
            if opt.local_rank == 0:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    '[lateral-3: {:.4f}]'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                            loss_record3.show()))

        if opt.local_rank == 0:
            logging.info(f"####################################Testing_EPOCH{epoch}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        total_dice = 0.0
        total_images = 0
        # for dataset in ['kvasir-Selected_0.909','CVC-ClinicDB-Selected', 'CVC-ColonDB', 'CVC-300',  'ETIS-LaribPolypDB', 'polygen']:
        # for dataset in [ 'Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen']:
        for dataset in [ 'Kvasir','CVC-ClinicDB']:

            metrics_result, num_images = valid(model, dataset,opt)
            total_dice += metrics_result['F1'] * num_images
            total_images += num_images
            if opt.local_rank == 0:
                print(f'TrainingEpoch[{epoch}]:\tTested on Dataset:[{dataset}]>>>>>>>>>')
                print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'], metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'], metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))
                mdice,miou,mprecision,mrecall=metrics_result['F1'], metrics_result['IoU_poly'], metrics_result['precision'],metrics_result['recall'],
                logging.info(f'TrainingEpoch[{epoch}]_Tesed_on_[{dataset}]:\tDice:{mdice}\tIoU:{miou}\tPrecision:{mprecision}\tRecall:{mrecall}')  

        if opt.local_rank == 0:
            meandice = total_dice/total_images
            if meandice > best:
                best_idx = epoch
                best = meandice
                torch.save(model.state_dict(), f'{opt.record_path}/{opt.dataset}_{epoch}_Best.pt')
            print(">>>>>>>>>>>>>>>>>Epoch %d with best mdice: %.4f" % (best_idx, best))
            logging.info(">>>>>>>>>>>>>>>>>Epoch %d with best mdice: %.4f" % (best_idx, best))

    if opt.local_rank == 0:
        torch.save(model.state_dict(), f'{opt.record_path}/{opt.dataset}_Last.pt')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=150, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=128, help='training batch size')
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

    opt = parser.parse_args()

    # ---- build models ----
    setup_seed(opt.seed)

    torch.cuda.set_device(opt.local_rank)
    dist.init_process_group(backend='nccl', )

    if opt.local_rank == 0:
        timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
        opt.record_path = f'./logs/{opt.dataset}/{timestamp}_{opt.tag}/'
        os.makedirs(opt.record_path,exist_ok=True)
        opt.log_path = opt.record_path + 'train_log.log'
        logging.basicConfig(filename=opt.log_path,
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = C2FNet().cuda()

    if opt.load_pretrained:
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

    params = model.parameters()
    optimizer = AdaXW(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(opt, image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    if opt.local_rank == 0:
        print("Start Training")

    train(train_loader, model, optimizer)
