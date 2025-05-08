import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
import csv

from lib.C2FNet import C2FNet
from utils.dataloader import test_dataset
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from tqdm import tqdm
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
    parser.add_argument('--pt_path', type=str, default='./EndoKED_SEG_Baselines/C2FNet-master/logs/Train_on_KvasirandDB/2024-04-21-04-29-16_loaded_from_endoked/Train_on_KvasirandDB_15_Best.pt')
    parser.add_argument('--testdata_path', type=str, default='./data/polyp_public/TestDataset', help='testing size')
    parser.add_argument('--exp_tag', type=str, default='debug', help='testing size')

    opt = parser.parse_args()
    
    #torch.cuda.set_device(0)  # set your gpu device
    pt_name = os.path.basename(opt.pt_path)
    save_root = opt.pt_path.split(pt_name)[0]

    opt.log_path = f'{save_root}/{opt.exp_tag}.log'
    logging.basicConfig(filename=opt.log_path,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = C2FNet()
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
    total_dice = 0.0
    total_num = 0
    total_fps = 0
    dataset_lst = ['Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen_all', 'polygen_videos_polyp_only','polygen_all_C1-6_multiSize']
    for _data_name in dataset_lst:
    # for _data_name in [ 'Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen_all', 'polygen_videos_polyp_only']:

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
        DSC = 0.0
        JACARD = 0.0
        preds = []
        gts = []

        dice_ls = []
        IoU = []
        precision = []
        recall = []
        hd_95 = []

        names = []

        time_all = 0
        for i in tqdm(range(num1)):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            
            time_start = time.time()
            res = model(image)
            time_end = time.time()
            time_all += (time_end - time_start)


            res = F.upsample(res, size=gt.shape[2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)        
            input = np.where(res >= 0.5, 1, 0)
            target = np.where(np.array(gt) >= 0.5, 1, 0)
            
            preds.append(input)
            gts.append(gt)
            
            N = gt.shape
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            union = input_flat + target_flat - intersection
            
            jacard = ((np.sum(intersection)+smooth)/(np.sum(union)+smooth))
            jacard = '{:.4f}'.format(jacard)
            jacard = float(jacard)
            JACARD += jacard
            
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC += dice

            dice_ls.append(f1_score(target_flat,input_flat))
            IoU.append(jaccard_score(target_flat,input_flat))
            precision.append(precision_score(target_flat,input_flat))
            recall.append(recall_score(target_flat,input_flat))
            names.append(name)

            if target.sum() > 0:
                if len(target.shape) > 2:
                    target = target.reshape(-1,target.shape[-2],target.shape[-1])[0]
                hd_95.append(hd95(input,target))
        
        total_dice += np.sum(dice_ls) 
        total_num += num1
        total_fps += (num1 / (time_all) )

        fps = num1 / (time_all) 

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

        logging.info(
                "\rTest:\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}, hd_95={:.6f}, fps={:.6f}".format(
                    np.mean(dice_ls),
                    np.mean(IoU),
                    np.mean(precision),
                    np.mean(recall),
                    np.mean(hd_95),
                    fps
                )
            )

        print('*****************************************************')
        print('Dice Score: ' + str(DSC/num1))
        print('Jacard Score: ' + str(JACARD/num1))
        
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