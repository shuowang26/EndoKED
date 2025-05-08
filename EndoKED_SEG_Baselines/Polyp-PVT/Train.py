import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging

import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import random
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
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1  = model(image)
        # eval Dice
        res = F.upsample(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25] 
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2= model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2 
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        # ---- train visualization ----
        if opt.local_rank == 0:
            if i % 20 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    ' lateral-5: {:0.4f}]'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                            loss_P2_record.show()))
    # save model 
    if opt.local_rank == 0:
        save_path = (opt.save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    #     torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    # choose the best model

    global dict_plot
   
    test1path = opt.test_path
    if (epoch + 1) % 1 == 0:
        total_dice=0.0
        total_num=0
        for dataset in ['CVC-ClinicDB', 'Kvasir']:
        # for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB','polygen_all']:
            dataset_dice, num = test(model, test1path, dataset)
            total_dice += dataset_dice * num
            total_num += num
            if opt.local_rank == 0:
                logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
                print(dataset, ': ', dataset_dice)
            # dict_plot[dataset].append(dataset_dice)
        meandice = total_dice / total_num
        # dict_plot['test'].append(meandice)
        if meandice > best:
            best = meandice
            if opt.local_rank == 0:
                # torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
                torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT-best.pth')
                print('##############################################################################best', best)
                logging.info('##############################################################################best:{}'.format(best))

def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[], 'polygen':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test', 'polygen']
    ##################model_name#############################
    model_name = 'PolypPVT'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=64, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')
    
    parser.add_argument('--dataset', type=str,
                        default='Train_on_KvasirandDB', help='Train_on_ZhongshanandKvasirandDB')

    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
    parser.add_argument("--gpu_number", default=1, type=int, help="local_rank")
    parser.add_argument('--seed', type=int,
                        default=0, help='gradient clipping margin')

    parser.add_argument('--tag', type=str,
                        default='scratch', help='decay rate of learning rate')
    parser.add_argument('--cpt_path', type=str,
                        default='./pretrained.pth', help='decay rate of learning rate')
    parser.add_argument('--load_pretrained',
                        default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="log tb")


    opt = parser.parse_args()
    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    setup_seed(opt.seed)

    torch.cuda.set_device(opt.local_rank)
    dist.init_process_group(backend='nccl', )

    model = PolypPVT().cuda()

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
        timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
        opt.record_path = f'./logs/{opt.dataset}/{timestamp}_{opt.tag}/'
        os.makedirs(opt.record_path,exist_ok=True)
        opt.log_path = opt.record_path + 'train_log.log'
        logging.basicConfig(filename=opt.log_path,
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

        opt.save_path = opt.record_path


    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)
    if opt.local_rank == 0:
        print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(opt, image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    if opt.local_rank == 0:
        print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
    
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)
