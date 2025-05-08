import numpy as np
import random
import multiprocessing

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data

from Data.dataset import SegDataset
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist



def split_ids(len_ids):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices


def get_dataloaders(input_paths, target_paths, val_img_path=None, val_mask_path=None, batch_size=8, local_rank=0):

    transform_input4train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((352, 352)), transforms.Grayscale()]
    )

    train_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4train,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
        affine=True,
    )

    test_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    if val_img_path is not None:
        val_dataset = SegDataset(
            input_paths=val_img_path,
            target_paths=val_mask_path,
            transform_input=transform_input4test,
            transform_target=transform_target,
        )
    else:
        val_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    train_indices, test_indices, val_indices = split_ids(len(input_paths))

    # train_dataset = data.Subset(train_dataset, train_indices)
    # val_dataset = data.Subset(val_dataset, val_indices)
    test_dataset = data.Subset(test_dataset, test_indices)

    # train_dataloader = data.DataLoader(
    #     dataset=train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=multiprocessing.Pool()._processes,
    # )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = data.DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  num_workers=multiprocessing.Pool()._processes,  
                                  pin_memory=True)
    

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=multiprocessing.Pool()._processes,
    )

    n_gpus = dist.get_world_size()
    split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]
    val_dataloader = data.DataLoader(
        dataset=split_dataset[local_rank],
        batch_size=1,
        shuffle=False,
        num_workers=multiprocessing.Pool()._processes,
    )

    return train_dataloader, test_dataloader, val_dataloader



def get_val_dataloaders(val_img_path=None, val_mask_path=None, batch_size=8):
    
    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((352, 352)), transforms.Grayscale()]
    )


    val_dataset = SegDataset(
        input_paths=val_img_path,
        target_paths=val_mask_path,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=multiprocessing.Pool()._processes,
    )

    return val_dataloader, val_dataloader, val_dataloader


