import sys
import os
sys.path.append('./EndoKED_SEG_Baselines/FCBFormer/')
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn
import random

from utils.dataloader import get_loader, test_dataset
from Models import models
from Metrics import performance_metrics
from Metrics import losses
import logging
import datetime
from Data.load_zhongshan import load_data_from_zhnogshan
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F



def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss,args):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset) // args.gpu_number,
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset) // args.gpu_number,
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(args, dataset, model, device, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    data_path = os.path.join(args.test_path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)

    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)

    ############TO DO
    DSC = 0.0
    for batch_idx in range(num1):
        data, target, name = test_loader.load_data()
        target = np.asarray(target, np.float32)
        target /= (target.max() + 1e-8)

        data = data.cuda()
        output = model(data)
        output = F.interpolate(output , size=target.shape, mode='bilinear', align_corners=False)
        output = output.sigmoid().data.cpu().numpy().squeeze()
        output = (output - output.min()) / (output.max() - output.min() + 1e-8)

        input = output
        target = np.array(target)
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ######## rebuild loader
    image_root = '{}/images/'.format(args.train_path)
    gt_root = '{}/masks/'.format(args.train_path)
    train_dataloader = get_loader(args, image_root, gt_root, batchsize=args.batch_size, trainsize=args.trainsize,
                            augmentation=args.augmentation)

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()

    perf = performance_metrics.DiceScore()

    model = models.FCBFormer()
    model.to(device)

    if args.load_pretrained:
        print("LOADING CPT from ENDOKED")
        cpt = torch.load(args.cpt_path,map_location='cpu')['model_state_dict']
        new = {}
        for k_,v in cpt.items():
            if 'module.' in k_:
                k = k_.replace('module.','')
                new[k] = v
        model.load_state_dict(new)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    return (
        device,
        train_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
    )


def train(args):
    (
        device,
        train_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
    ) = build(args)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = None
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss,args
            )

            ##################TEST 
            total_dice=0.0
            total_num=0
            for dataset in ['CVC-ClinicDB', 'Kvasir']:
                test_measure_mean, num = test(args, dataset,
                    model, device, epoch, perf
                )
                total_dice += test_measure_mean * num
                total_num += num

                if args.local_rank == 0:
                    logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, test_measure_mean))
                    print(dataset, ': ', test_measure_mean)
            ##################TEST 
            mdice = total_dice / total_num

        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)

        if args.lrs == "true":
            scheduler.step(mdice)
        if prev_best_test == None or mdice > prev_best_test:
            if args.local_rank == 0:
                logging.info(f'########epoch:[{epoch}]:\t best_dice:{test_measure_mean}########')
                print(f'########epoch:[{epoch}]:\t best_dice:{test_measure_mean}########')
                print("Saving...")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "test_measure_mean": test_measure_mean,
                    },
                    args.record_path + args.dataset + f"{epoch}_Best.pt",
                )
            prev_best_test = mdice
        
    torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "test_measure_mean": test_measure_mean,
            },
            args.record_path + args.dataset + "_LastEpoch.pt",
            )
    
def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, default="Train_on_ZhongshanandKvasirandDB")
    parser.add_argument("--train_path", type=str, default="/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TrainDataset/")
    parser.add_argument("--test_path", type=str, default="/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/test/")
    parser.add_argument("--log_path", type=str, default='./train_log.log', dest="val_root")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')         
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
    parser.add_argument("--gpu_number", default=1, type=int, help="local_rank")
    parser.add_argument('--seed', type=int,
                        default=0, help='gradient clipping margin')
    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--tag', type=str,
                        default='scratch', help='decay rate of learning rate')
    parser.add_argument('--cpt_path', type=str,
                        default='./pretrained.pth', help='decay rate of learning rate')
    parser.add_argument('--load_pretrained',
                        default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="log tb")


    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = get_args()
    setup_seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', )

    if args.local_rank == 0:
        timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now())
        args.record_path = f'./logs/{args.dataset}/{timestamp}_{args.tag}/'
        os.makedirs(args.record_path,exist_ok=True)
        args.log_path = args.record_path + 'train_log.log'
        logging.basicConfig(filename=args.log_path,
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    train(args)

if __name__ == "__main__":
    main()
