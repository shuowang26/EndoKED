import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn

from Data import dataloaders_endo as dataloaders
from Models import models
from Metrics import performance_metrics

import warnings
warnings.filterwarnings("ignore")


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    val_img_path = args.val_root + "images/*"
    val_input_paths = sorted(glob.glob(val_img_path))
    val_depth_path = args.val_root + "masks/*"
    val_target_paths = sorted(glob.glob(val_depth_path))

    train_dataloader, _, test_dataloader = dataloaders.get_val_dataloaders(
        val_input_paths, val_target_paths, batch_size=1
    )

    perf = performance_metrics.DiceScore()

    model = models.FCBFormer()

    state_dict = torch.load(
        args.pt_path
    )

    new_dict = {}
    for k,v in state_dict["model_state_dict"].items():
        if 'module.' in k:
            k_ = k.replace('module.','')
            new_dict[k_] = v
        else:
            new_dict[k] = v

    model.load_state_dict(new_dict)

    model.to(device)

    return device, test_dataloader, perf, model, val_target_paths


@torch.no_grad()
def predict(args):
    device, test_dataloader, perf_measure, model, target_paths = build(args)
    save_path = f"{args.mask_save_path}/Predictions/Trained_on {args.train_dataset}/Tested_on_{args.test_dataset}"
    os.makedirs(save_path,exist_ok=True)

    t = time.time()
    model.eval()
    perf_accumulator = []
    for i, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
        item = os.path.basename(target_paths[i])
        img_save_path = f"{save_path}/{item}"
        cv2.imwrite(
            img_save_path,
            predicted_map * 255,
        )
        if i + 1 < len(test_dataloader):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument(
        "--train_dataset", type=str, default='Train_on_KvasirandDB'
    )
    parser.add_argument("--pt_path", default='/home/gpu_user/data/yzw/endokd_rebuttal/FCBFormer/logs/Train_both/Train_on_KvasirandDB/Train_on_KvasirandDB.pt')

    return parser.parse_args()


def main():
    # dataset=['polygen','CVC-300', 'CVC-ClinicDB-Selected', 'kvasir-Selected_0.909', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    dataset=['polygen','CVC-300', 'CVC-ClinicDB', 'kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    args = get_args()
    for data_name in dataset:
        args.test_dataset = data_name
        args.val_root = f'/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/{data_name}/'
        basename = os.path.basename(args.pt_path)
        args.mask_save_path = args.pt_path.split(f'{basename}')[0]
        print(f'Generate masks for dataset [{data_name}]....')
        predict(args)

if __name__ == "__main__":
    main()

