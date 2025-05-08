import csv
from glob import glob


def load_data_from_zhnogshan():
    root = './data/poly_detection'
    img_path = f'{root}/images/*'
    mask_path = f'{root}/masks/*'

    img_path_list, label_path_lst = sorted(glob(img_path)), sorted(glob(mask_path))

    return img_path_list, label_path_lst