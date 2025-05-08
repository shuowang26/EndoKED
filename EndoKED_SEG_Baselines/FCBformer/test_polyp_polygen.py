import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
import csv

from Models import models
from utils.dataloader import test_dataset
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import glob
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


def build(args,data_path):
    val_input_paths, val_target_paths = None, None
    val_img_path = data_path + "/images/*"
    val_input_paths = sorted(glob.glob(val_img_path))
    val_depth_path = data_path + "/masks_renamed/*"
    val_target_paths = sorted(glob.glob(val_depth_path))

    train_dataloader, _, val_dataloader = dataloaders.get_val_dataloaders(
        val_input_paths, val_target_paths, batch_size=1
    )
    return val_dataloader, val_target_paths,


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pt_path', type=str, default='./EndoKED_SEG_Baselines/FCBFormer/logs/Train_on_KvasirandDB/2024-04-22-06-05-05_loaded_from_endoked/Train_on_KvasirandDB30_Best.pt')
    parser.add_argument('--testdata_path', type=str, default='/data/Datasets/息肉数据集/息肉公开数据集/dataset/TestDataset/', help='testing size')
    parser.add_argument("--root", type=str, default="/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TrainDataset/")
    parser.add_argument('--exp_tag', type=str, default='debug', help='testing size')

    opt = parser.parse_args()
    
    #torch.cuda.set_device(0)  # set your gpu device
    pt_name = os.path.basename(opt.pt_path)
    save_root = opt.pt_path.split(pt_name)[0]

    opt.log_path = f'{save_root}/{opt.exp_tag}.log'
    logging.basicConfig(filename=opt.log_path,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = models.FCBFormer()
    model.cuda()

    cpt = torch.load(opt.pt_path)
    new_dict = {}
    for k,v in cpt["model_state_dict"].items():
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
    # for _data_name in ['data_C1','data_C2','data_C3','data_C4','data_C5','data_C6']:
    # for _data_name in [ 'Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen_all']:
    # for _data_name in ['Kvasir','polygen_videos_polyp_only']:
    # for _data_name in [ 'Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen_all', 'polygen_videos_polyp_only']:
    dataset_lst = ['Kvasir','CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB','polygen_all', 'polygen_videos_polyp_only', 'polygen_all_C1-6_multiSize']
    for _data_name in dataset_lst:


        ##### put data_path here #####
        data_path = f'{opt.testdata_path}/{_data_name}/'
        
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

        hd_95 = []

        IoU = []
        precision = []
        recall = []
        dice_ls = []
        names = []



        ############TO DO
        DSC = 0.0
        time_all = 0
        for batch_idx in tqdm(range(num1)):
            data, target, name = test_loader.load_data()
            target = np.asarray(target, np.float32)
            target /= (target.max() + 1e-8)

            data = data.cuda()
            time_start = time.time()
            output = model(data)
            time_end = time.time()
            time_all += (time_end - time_start)

            output = F.interpolate(output , size=target.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            cv2.imwrite(save_path+name, output*255)        


            input = output
            target = np.array(target)
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            union = input_flat + target_flat - intersection

            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC = DSC + dice

            if target.sum() > 0:
                if len(target.shape) > 2:
                    target = target.reshape(-1,target.shape[-2],target.shape[-1])[0]
                input = np.where(input >= 0.5, 1, 0)
                target = np.where(np.array(target) >= 0.5, 1, 0)
                hd_95.append(hd95(input,target))

            input_flat = np.where(input_flat >= 0.5, 1, 0)
            target_flat = np.where( target_flat>= 0.5, 1, 0)
            IoU.append(jaccard_score(target_flat,input_flat))
            precision.append(precision_score(target_flat,input_flat))
            recall.append(recall_score(target_flat,input_flat))
            names.append(name)
            dice_ls.append(f1_score(target_flat,input_flat))


        total_dice += DSC
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
        hdmean = np.mean(hd_95)

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
        print('Dice Score: ' + str(DSC/num1)+ 'HD95: ' + str(hdmean) + 'FPS: ' + str(fps))

        logging.info('Dice Score: ' + str(DSC/num1)+ 'HD95: ' + str(hdmean) + 'FPS: ' + str(fps))
        
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
    macs, params = profile(model, inputs=(data, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(f'MACs:{macs}')
    print(f'Params:{params}')

    logging.info(
            f"***************Model Info*****************\
            MACs:{macs},Params:{params} "
        )