import os
import glob
import argparse
import numpy as np

from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from skimage.io import imread
from skimage.transform import resize

from Data.dataloaders import split_ids
import logging
import warnings
warnings.filterwarnings("ignore")


def eval(args):

    mask_base = f"{args.mask_save_path}/Predictions/Trained_on {args.train_dataset}/Tested_on_{args.test_dataset}/*"
    prediction_files = sorted(
        glob.glob(
            mask_base
        )
    )

    depth_path = args.val_root + "masks/*"
    target_paths = sorted(glob.glob(depth_path))
    
    test_files = target_paths

    dice = []
    IoU = []
    precision = []
    recall = []

    for i in range(len(test_files)):
        pred = np.ndarray.flatten(imread(prediction_files[i]) / 255) > 0.5
        gt = (
            resize(imread(test_files[i]), (int(352), int(352)), anti_aliasing=False)
            > 0.5
        )

        if len(gt.shape) == 3:
            gt = np.mean(gt, axis=2)
        gt = np.ndarray.flatten(gt)

        dice.append(f1_score(gt, pred))
        IoU.append(jaccard_score(gt, pred))
        precision.append(precision_score(gt, pred))
        recall.append(recall_score(gt, pred))

        if i + 1 < len(test_files):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}".format(
                    i + 1,
                    len(test_files),
                    100.0 * (i + 1) / len(test_files),
                    np.mean(dice),
                    np.mean(IoU),
                    np.mean(precision),
                    np.mean(recall),
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}".format(
                    i + 1,
                    len(test_files),
                    100.0 * (i + 1) / len(test_files),
                    np.mean(dice),
                    np.mean(IoU),
                    np.mean(precision),
                    np.mean(recall),
                )
            )
    logging.info(
            "\rTest: [{}/{} ({:.1f}%)]\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}".format(
                i + 1,
                len(test_files),
                100.0 * (i + 1) / len(test_files),
                np.mean(dice),
                np.mean(IoU),
                np.mean(precision),
                np.mean(recall),
            )
        )




def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument(
        "--train_dataset", type=str, default='Train_on_KvasirandDB'
    )
    parser.add_argument("--pt_path", default='/home/gpu_user/data/yzw/endokd_rebuttal/FCBFormer/logs/Train_both/2024-03-14-22-56-35/Train_both.pt')

    return parser.parse_args()


def main():
    # dataset=['kvasir-Selected_0.909','CVC-ClinicDB-Selected', 'CVC-ColonDB', 'CVC-300',  'ETIS-LaribPolypDB', 'polygen']
    dataset=['polygen','CVC-300', 'CVC-ClinicDB', 'kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    args = get_args()
    basename = os.path.basename(args.pt_path)
    args.mask_save_path = args.pt_path.split(f'{basename}')[0]
    args.log_path = args.mask_save_path + 'eval_results.log'
    logging.basicConfig(filename=args.log_path,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p')

    for data_name in dataset:
        args.test_dataset = data_name
        args.val_root = f'/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/{data_name}/'
        print(f'Evaluate masks for dataset [{data_name}]....')
        logging.info(f'Evaluate masks for dataset [{data_name}]....')
        eval(args)

if __name__ == "__main__":
    main()