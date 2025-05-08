import cv2
from glob import glob 
import os
from tqdm import tqdm 


def conver(root_dir):
    gt_root = f"{root_dir}/masks/"
    img_root = f"{root_dir}/images/*"
    images_ls = sorted(glob(img_root))

    save_root = f'{root_dir}/masks_renamed/'
    os.makedirs(save_root,exist_ok=True)

    for item in tqdm(images_ls):
        image_name = os.path.basename(item)
        gt_name = image_name.replace('.jpg','_mask.jpg')
        gt_path = f"{gt_root}/{gt_name}"
        save_path = f'{save_root}/{image_name}'
        gt = cv2.imread(gt_path)
        cv2.imwrite(save_path, gt)
    
    return
        

if __name__ == "__main__":
    train_data_base = '/home/gpu_user/data/EndoGPT/database/息肉公开数据集/dataset/TestDataset/Polygen_all_frames/'
    for _data_name in [ 'data_C1','data_C2','data_C3','data_C4','data_C5','data_C6']:
        print(f">>>>>>>>>>>Rename {_data_name}")
        root_dir = f'{train_data_base}/{_data_name}/'
        conver(root_dir)

