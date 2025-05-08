import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import datasets
from utils.metrics import evaluate
from opt import opt
from utils.comm import generate_model
from utils.loss import DeepSupervisionLoss
from utils.metrics import Metrics
import os
import torch.nn.functional as F
import logging 
from datetime import datetime
import numpy as np
import random 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
import warnings
warnings.filterwarnings('ignore')


def valid(model, dataset, opt):

    opt.valid_data_dir = f'{opt.val_root}/{dataset}/'
    valid_data = getattr(datasets, "PolypDataset")(opt, opt.valid_data_dir, mode='valid')
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch = int(len(valid_data) / 1)
    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2', 'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    with torch.no_grad():
        bar = tqdm(enumerate(valid_dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)

            _recall, _specificity, _precision, _F1, _F2, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, IoU_bg= _IoU_bg, IoU_mean= _IoU_mean)

    metrics_result = metrics.mean(total_batch)

    return metrics_result, total_batch


def train():

    # load model
    if opt.local_rank == 0:
        print('Loading model......')
    model = generate_model(opt).cuda()

    if opt.load_pretrained:
        print("LOADING CPT from ENDOKED")
        cpt = torch.load(opt.cpt_path,map_location='cpu')
        new = {}
        for k_,v in cpt.items():
            if 'module.' in k_:
                k = k_.replace('module.','')
                new[k] = v
        model.load_state_dict(new)

    model = DistributedDataParallel(model, device_ids=[opt.local_rank], find_unused_parameters=True)

    if opt.local_rank == 0:
        print('Load model:', opt.model)

        # load data
        print('Loading data......')

    train_data = getattr(datasets, "PolypDataset")(opt,opt.train_data_dir, mode='train')
    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=opt.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  num_workers=multiprocessing.Pool()._processes,  
                                  pin_memory=True)

    # load optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    lr_lambda = lambda epoch: pow(1.0 - epoch / opt.nEpoch, opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # train
    if opt.local_rank == 0:

        print('Start training')
        print('---------------------------------\n')
        
    best = 0
    best_idx = 0
    for epoch in range(opt.nEpoch):
        if opt.local_rank == 0:
            print('------ Epoch', epoch + 0)
        model.train()
        total_batch = int(len(train_data) / opt.batch_size / opt.gpu_num) 
        bar = tqdm(enumerate(train_dataloader), total=total_batch)
        
        for i, data in bar:
            img = data['image']
            gt = data['label']
        
            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            optimizer.zero_grad()
            output = model(img)
            loss = DeepSupervisionLoss(output, gt)
            loss.backward()

            optimizer.step()
            bar.set_postfix_str('loss: %.5s' % loss.item())

        scheduler.step()

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
                logging.info(f'TrainingEpoch[{epoch}]:Tested on Dataset:[{dataset}]\tDice:{mdice}\tIoU:{miou}\tPrecision:{mprecision}\tRecall:{mrecall}')  

        if opt.local_rank == 0:
            meandice = total_dice/total_images
            if meandice > best:
                best = meandice
                best_idx = epoch
                torch.save(model.state_dict(), f'{opt.record_path}/{opt.dataset}_{epoch}_Best.pt')
                print(">>>>>>>>>>>>>>>>>Epoch %d with best mdice: %.4f" % (best_idx, best))
                logging.info(">>>>>>>>>>>>>>>>>Epoch %d with best mdice: %.4f" % (best_idx, best))
        
    if opt.local_rank == 0:
        print(">>>>>>>>>>>>>>>>>Epoch %d with best mdice: %.4f" % (best_idx, best))
        logging.info(">>>>>>>>>>>>>>>>>>>>>Epoch %d with best mdice: %.4f" % (best_idx, best))
        torch.save(model.state_dict(), f'{opt.record_path}/{opt.dataset}_Last.pt')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':

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

    
    if opt.mode == 'train':
        if opt.local_rank == 0:
            print('---PolypSeg Train---')
        train()
    

