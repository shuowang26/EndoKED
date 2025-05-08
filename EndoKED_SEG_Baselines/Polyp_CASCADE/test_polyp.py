import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2

from lib.networks import PVT_CASCADE
from utils.dataloader import test_dataset
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    method_name = 'PolypPVT-CASCADE'
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='/home/gpu_user/data/yzw/endokd_rebuttal/CASCADE-main/logs/Train_on_KvasirandDB/2024-03-23-19-17-38/PolypPVT-CASCADE_best.pth')
    parser.add_argument('--testdata_path', type=str, default='/data/Datasets/息肉数据集/息肉公开数据集/dataset/TestDataset/', help='testing size')

    opt = parser.parse_args()
    
    #torch.cuda.set_device(0)  # set your gpu device
    pt_name = os.path.basename(opt.pth_path)
    save_root = opt.pth_path.split(pt_name)[0]

        opt.log_path = f'{save_root}/{opt.exp_tag}.log'
    logging.basicConfig(filename=opt.log_path,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = PVT_CASCADE()
    model.cuda()

    cpt = torch.load(opt.pth_path)
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
    for _data_name in [ 'data_C3','data_C2','data_C1','data_C4','data_C5','data_C6']:
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

        for i in tqdm(range(num1)):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            
            res1, res2, res3, res4 = model(image) # forward
            
            # eval Dice
            res = F.upsample(res1 + res2 + res3 + res4, size=gt.shape, mode='bilinear', align_corners=False)

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
        
            total_dice += np.sum(dice_ls) 
            total_num += num1
        logging.info(
                "\rTest:\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}".format(
                    np.mean(dice_ls),
                    np.mean(IoU),
                    np.mean(precision),
                    np.mean(recall),
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