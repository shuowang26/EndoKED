import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
import cv2
from medpy import metric
import matplotlib.pyplot as plt

# calculate dice 
def calculate_metric_percase(pred, gt):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    # gt = gtnumpy()[:,0,...]
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
    
def cal_dice(outputs,label):
    # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
    # out = out.cpu().detach().numpy()
    out = outputs
    dice,hd95 = (calculate_metric_percase(out, label))

    return dice,hd95


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='/data/PROJECTS/Endo_GPT/reference_codes/Polyp-PVT-main/pretrained_pth/31PolypPVT-best.pth')
    opt = parser.parse_args()
    model = PolypPVT()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    # for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    for _data_name in ['test']:
    # for _data_name in ['CVC-ClinicDB']:
        dice_all = []

        ##### put data_path here #####
        # data_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/CVC-ClinicDB_datasets/CVC-ClinicDB_datasets/{}'.format(_data_name)
        data_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/dataset/TestDataset/{}'.format(_data_name)
        # data_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/{}'.format(_data_name)
        ##### save_path #####
        save_path = './result_map/PolypPVT_logit_selected/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            # gt = np.uint8(np.asarray(gt, np.float32) == 255)
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P1,P2 = model(image)
            res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res =(res - res.min()) / (res.max() - res.min() + 1e-8)
            input = res
            target = np.array(gt)
            N = gt.shape
            smooth = 1
            # input_flat = np.reshape(input, (-1))
            input_flat = np.reshape(np.uint8(input>=0.01), (-1))

            # target_flat = np.reshape(target, (-1))
            target_flat = np.reshape(np.uint8(target>=0.01), (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice_all.append(dice)
        
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(gt)
            # plt.subplot(122)
            # plt.imshow(res)
            # plt.title(f'dice:{dice:.4f}')_fa
            # plt.show()
            # plt.close()

            # cv2.imwrite(save_path+name, res*255)
        dice_all = np.stack(dice_all,axis=0)
        dice_final = dice_all.mean()
        print(_data_name, 'Finish!')
        print(f'dice:{dice_final}')