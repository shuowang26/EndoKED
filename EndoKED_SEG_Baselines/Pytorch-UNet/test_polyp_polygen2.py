import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from utils.metrics import Metrics
from utils.metrics import evaluate
import cv2
import csv

from unet import UNet
from utils.dataloader import test_dataset
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from glob import glob
import logging
import warnings
warnings.filterwarnings('ignore')
import time
from medpy.metric.binary import hd95 

def save_ds_as_csv(names,dices,save_path='./train.csv'):
    f = open(save_path,'w')
    writer = csv.writer(f,lineterminator='\n')
    header = ['frame_name','dice']
    writer.writerow(header)
    for name_item,dice_item in zip(names,dices):
        data_info = [name_item,dice_item]
        writer.writerow(data_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pt_path', type=str, default='/home/gpu_user/data/yzw/endokd_rebuttal/CASCADE-main/logs/Train_on_KvasirandDB/2024-03-23-19-17-38/PolypPVT-CASCADE_best.pth')
    parser.add_argument('--testdata_path', type=str, default='/data/Datasets/息肉数据集/息肉公开数据集/dataset/TestDataset/', help='testing size')
    parser.add_argument('--exp_tag', type=str, default='eval_on_frames', help='testing size')

    opt = parser.parse_args()
    
    #torch.cuda.set_device(0)  # set your gpu device
    pt_name = os.path.basename(opt.pt_path)
    save_root = opt.pt_path.split(pt_name)[0]

    opt.log_path = f'{save_root}/{opt.exp_tag}.log'
    logging.basicConfig(filename=opt.log_path,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model.cuda()

    cpt = torch.load(opt.pt_path)
    new_dict = {}
    for k,v in cpt.items():
        if 'module.' in k:
            k_ = k.replace('module.','')
            new_dict[k_] = v
        else:
            new_dict[k] = v

    model.load_state_dict(new_dict)
    model.eval()
    
    # for _data_name in ['kvasir-Selected_0.909','CVC-ClinicDB-Selected', 'CVC-ColonDB', 'CVC-300',  'ETIS-LaribPolypDB', 'polygen']:
    # for _data_name in [ 'Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen']:
    total_dice = 0.0
    total_num = 0
    total_fps = 0
    # for _data_name in [ 'Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen_all']:
    # for _data_name in ['Kvasir','polygen_videos_polyp_only']:
    # for _data_name in [ 'Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen_all', 'polygen_videos_polyp_only']:
    dataset_lst = ['Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen_all', 'polygen_videos_polyp_only','polygen_all_C1-6_multiSize']
    for _data_name in dataset_lst:
        ##### put data_path here #####
        data_path = f'{opt.testdata_path}/{_data_name}'
        
        ##### save_path #####
        save_path = f'{save_root}/Predictions/tes_on_{_data_name}/'

        logging.info(f'Evaluating[{_data_name}] now...')

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print('Evaluating ' + data_path)
        
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)

        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        total_batch = int(len(os.listdir(gt_root)) / 1)


        metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2', 'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])
        hd_95 = []
        dice_ls = []
        names = []

        bar = tqdm(test_loader.images)
        time_all = 0
        for i in bar:
            image, gt, name = test_loader.load_data()
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            gt = gt.cuda()
            time_start = time.time()
            pred = model(image)
            time_end = time.time()

            time_all += (time_end - time_start)

            output = pred[len(pred)-1].sigmoid()

            _recall, _specificity, _precision, _F1, _F2, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, IoU_bg= _IoU_bg, IoU_mean= _IoU_mean)
            dice_ls.append(_F1)
            names.append(name)


            res = output.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)      

            output = np.where(output.detach().cpu().numpy() >= 0.5, 1, 0)
            gt = np.where(gt.detach().cpu().numpy() >= 0.5, 1, 0)
            
            if gt.sum() > 0 and output.sum() > 0:
                if len(gt.shape) > 2:
                    gt = gt.reshape(-1,gt.shape[-2],gt.shape[-1])[0]
                hd_95.append(hd95(output[0,...], gt)) 

        metrics_result = metrics.mean(total_batch)
        dice = metrics_result['F1']
        IoU = metrics_result['IoU_mean']
        precision = metrics_result['precision']
        recall = metrics_result['recall']

        total_dice += dice * total_batch
        total_num += total_batch
        total_fps += (total_batch / (time_all) )
        fps = total_batch / time_all

        csv_path = f'{save_root}/dice_lists/{_data_name}_dice_ls.csv'
        csv_root = f'{save_root}/dice_lists/'
        if not os.path.exists(csv_root):
            os.makedirs(csv_root)
        save_ds_as_csv(names,dice_ls,save_path=csv_path)

        csv_path = f'{save_root}/hd95_lists/{_data_name}_hd95_ls.csv'
        csv_root = f'{save_root}/hd95_lists/'
        if not os.path.exists(csv_root):
            os.makedirs(csv_root)
        save_ds_as_csv(names,hd_95,save_path=csv_path)

        # time_all = 0
        # total_batch = 0

        logging.info(
                "\rTest:\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}, hd_95={:.6f},FPS={:.6f}".format(
                    (dice),
                    (IoU),
                    (precision),
                    (recall),
                    np.mean(hd_95),
                    (fps)
                )
            )
            
        print('*****************************************************')
        print('Dice Score: ' + str(dice))
        
        print(_data_name, 'Finish!')
        print('*****************************************************')


    mean_dice = total_dice / total_num

    mean_fps = total_fps / len(dataset_lst)


    logging.info(
            f"***************Total average Dice*****************\
            MEAN_DICE:{mean_dice}"
        )
    logging.info(
            f"***************Total average FPS*****************\
            MEAN_FPS:{mean_fps}"
        )
    
    print(
            f"***************Total average Dice*****************\
            MEAN_DICE:{mean_dice}"
        )

    print(
            f"***************Total average FPS*****************\
            MEAN_FPS:{mean_fps}"
        )

    from thop import profile
    from thop import clever_format
    macs, params = profile(model, inputs=(image, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(f'MACs:{macs}')
    print(f'Params:{params}')

    logging.info(
            f"***************Model Info*****************\
            MACs:{macs},Params:{params} "
        )