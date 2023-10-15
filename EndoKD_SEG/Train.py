import torch
from torch.autograd import Variable
import os
import csv
from glob import glob
import argparse
from lib.pvt import PolypPVT
from utils.dataloader_zhongshan import get_loader, test_dataset, denormalize_img
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import random 
import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset, epoch,normalize):
    if opt.local_rank == 0:
        writer = SummaryWriter(opt.tb_path)
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352, normalize)
    DSC = 0.0
    images = []
    gts = []
    preds = []
    rand_index = np.random.randint(0,(num1 - opt.batchsize))
    resize = transforms.Resize((352, 352))

    for i in range(num1):
        image, gt_, name = test_loader.load_data()
        if normalize:
            image_denorm = denormalize_img(image)
        else:
            image_denorm = image

        gt = np.asarray(gt_, np.float32)
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

        ### visual val 
        visual_num = 12
        if i >= rand_index and (i-rand_index) < visual_num:
            images.append(image_denorm[0])
            gt = resize(gt_)
            input = resize(Image.fromarray(np.array(input)))
            gts.append(torch.tensor(np.array(gt)[None,...]))
            preds.append(torch.tensor(np.array(input)[None,...]))

        if (i-rand_index) == visual_num and opt.local_rank == 0:
            images = torch.stack(images,dim=0)
            gts = torch.stack(gts,dim=0)
            preds = torch.stack(preds,dim=0)
            grid_image  = torchvision.utils.make_grid(images,4)
            grid_gt  = torchvision.utils.make_grid(gts.clone(),4)
            grid_pred  = torchvision.utils.make_grid(preds.clone(),4)
            writer.add_image(f"val_{dataset}/images",grid_image,global_step = epoch)
            writer.add_image(f"val_{dataset}/gt",grid_gt,global_step = epoch)
            writer.add_image(f"val_{dataset}/pred",grid_pred,global_step = epoch)
    
    return DSC / num1

def train(train_loader, model, optimizer, epoch, test_path, normalize,):
    if opt.local_rank == 0:
        writer = SummaryWriter(opt.tb_path)
    model.train()
    global best
    global test_info
    size_rates = [0.75, 1, 1.25] 
    loss_P2_record = AvgMeter()

    num = total_step // 20 
    tb_idx = 0 + epoch * (num)

    for i, pack in enumerate(train_loader, start=1):

        if i >= (total_step):
            break

        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            if normalize:
                image_denorm = denormalize_img(images)
            else:
                image_denorm = images

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
            if opt.local_rank == 0:
                writer.add_scalars('train/loss',{'total_loss':loss.item(),'P1_loss':loss_P1.item(),\
                                    'P2_loss':loss_P1.item()},global_step=(epoch*total_step + i))
                ### visual
                visual_num = 12
                visual_nums = np.random.randint(opt.batchsize,size=(1,visual_num))[0]
                if i % 20 == 0:
                    res = F.upsample(P1 + P2 , size=gts.shape[2:], mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    input = res[:,None,...]
                    grid_image = torchvision.utils.make_grid(image_denorm[visual_nums,...],4)
                    grid_pred  = torchvision.utils.make_grid(torch.tensor(input[visual_nums,...]).clone(),4)
                    grid_gt  = torchvision.utils.make_grid(torch.tensor(gts[visual_nums,...]).clone(),4)
                    writer.add_image(f"train/images",grid_image,global_step = tb_idx)
                    writer.add_image(f"train/gt", grid_gt,global_step = tb_idx)
                    writer.add_image(f"train/preds", grid_pred,global_step= tb_idx)
                    tb_idx += 1

            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            if opt.local_rank == 0:
                logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    ' lateral-5: {:0.4f}]'.
                    format(datetime.datetime.now(), epoch, opt.epoch, i, total_step,
                            loss_P2_record.show()))
    # save model 
    if opt.local_rank == 0:
        save_path = (opt.train_save)
        torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    # choose the best model

    global dict_plot
   
    if (epoch + 1) % 1 == 0:
        for dataset in dataset_lst:
            dataset_dice = test(model, test_path, dataset, epoch, normalize)
            if opt.local_rank == 0:
                logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
                print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)

            if opt.local_rank == 0:
                writer.add_scalars(f'val_{dataset}/dice',{f'{dataset}':dataset_dice},global_step=epoch)

            if test_info[dataset][0] <= dataset_dice:
                best_info = [dataset_dice,epoch]
                test_info[dataset] = best_info

        if opt.local_rank == 0:
            logging.info(f'############################### dataset best dice and iter###########################')
            logging.info(f'{test_info}')


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
    
    
    
if __name__ == '__main__':

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
                        default=24, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=200, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--tb_path', type=str,
                        default='./tb_path',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_Pred_woSAM_pth/')
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
    parser.add_argument("--log_file_name", default='all_train_log_Pred_woSAM.log', type=str, help="all_train_log.log")
    

    opt = parser.parse_args()
    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    setup_seed(0)
    torch.cuda.set_device(opt.local_rank)
    dist.init_process_group(backend='nccl', )
    model = PolypPVT().cuda()
    model = DistributedDataParallel(model, device_ids=[opt.local_rank], find_unused_parameters=True)

    best = 0


    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)



    """###### Dir Input #####"""
    normalize = False
    type = 'Preds' 
    # type = 'CAM' 
    if opt.local_rank == 0:
        timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
        log_root = f'/data/PROJECTS/Endo_GPT/reference_codes/Polyp-PVT-main/logs/{type}/{timestamp}'
        opt.log_file_name = f'{log_root}/filterd_imgs_from_{type}.log'
        opt.train_save = f'{log_root}/model_{type}_pth/'
        opt.tb_path = f'{log_root}/tensorboard_log_all_from_{type}'
        os.makedirs(log_root,exist_ok=True)
        os.makedirs(opt.tb_path,exist_ok=True)
        os.makedirs(opt.train_save,exist_ok=True)
        logging.basicConfig(filename=opt.log_file_name,
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        
    img_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/datasets/a_zhongshan_2wpublic_all/filtered_decoder_after_Preds_zhongshan_2wpublic_imgPath.csv'
    label_root_path = '/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/0629_Preds_seeds_decoder/'
    test_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/dataset/TestDataset/'
    img_path_list, label_name_lst = [], []

    dataset_lst = ['CVC-300', 'CVC-ClinicDB-Selected', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB','test']
    test_info = {}
    dict_plot = {}
    for key in dataset_lst:
        test_info[key] = [0,-1]
        dict_plot[key] = []
    name = dataset_lst

    with open (img_path,'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for item in reader:
            img_path_list.append(item[0])
            label_path = label_root_path + item[1]
            label_name_lst.append(label_path)

    train_loader = get_loader(img_path_list, label_name_lst, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation, normalize=normalize)

    ## select iter number instead of epoch
    # total_step = len(train_loader)
    total_step = len(train_loader) // 20
    if opt.local_rank == 0:
        print("#" * 20, "Start Training", "#" * 20)
        print(optimizer)
        logging.info("#####################optimizer")
        logging.info(f'{optimizer}\n')
        logging.info("#####################configs")
        logging.info(f'{opt}\n')
        logging.info("#####################load files")
        logging.info(f'image_path_load:{img_path}')
        logging.info(f'label_path_loaded:{label_root_path}')

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, test_path,normalize)
