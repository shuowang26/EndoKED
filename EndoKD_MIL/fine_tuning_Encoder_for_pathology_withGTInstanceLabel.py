import argparse
import warnings
import os
import time
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
# import models
from models.resnetv1 import resnet_CAMELYON_Encoder, teacher_Attention_head, student_head, teacher_DSMIL_head
from models.resnet_pretrained import PretrainedResNet18_Encoder, PretrainedResNet50_Encoder
from Datasets_loader.dataset_Endo_3Center_withPathologyLabel import gather_align_EndoImg_Center_forPathology, split_GT_pathology_alongPatient, gather_external_Pathology, Endo_img_MIL_all_center_withPathologyLabel
import datetime
import utliz
import util
import random
from tqdm import tqdm
import copy
import pandas as pd
from torchvision import datasets, transforms
import math


class Optimizer:
    def __init__(self, model_encoder, model_studentHead,
                 optimizer_encoder, optimizer_studentHead,
                 train_instanceloader,
                 test_instanceloader,
                 writer=None, num_epoch=100,
                 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model_encoder = model_encoder
        self.model_studentHead = model_studentHead
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_studentHead = optimizer_studentHead

        self.train_instanceloader = train_instanceloader
        self.test_instanceloader = test_instanceloader

        self.writer = writer
        self.num_epoch = num_epoch
        self.dev = dev
        self.log_period = 10

    def optimize(self):
        for epoch in range(self.num_epoch):
            self.optimize_student(epoch)
            self.evaluate_student(epoch, loader=self.test_instanceloader, log_prefix='test')
        return 0

    def optimize_student(self, epoch):
        self.model_encoder.train()
        self.model_studentHead.train()
        ## optimize teacher with instance-dataloader
        # 1. change loader to instance-loader
        loader = self.train_instanceloader
        # 2. optimize
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student training')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.squeeze().to(self.dev)

            feat = self.model_encoder(data)
            patch_prediction = self.model_studentHead(feat)
            patch_prediction = torch.softmax(patch_prediction, dim=1)

            # cal loss
            loss_student = -1. * torch.mean((1-label[0]) * torch.log(patch_prediction[:, 0] + 1e-5) +
                                             label[0] * torch.log(patch_prediction[:, 1] + 1e-5))
            self.optimizer_encoder.zero_grad()
            self.optimizer_studentHead.zero_grad()
            loss_student.backward()
            self.optimizer_encoder.step()
            self.optimizer_studentHead.step()

            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss', loss_student.item(), niter)

        instance_auc_ByStudent = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred.reshape(-1))
        self.writer.add_scalar('train_instance_AUC', instance_auc_ByStudent, epoch)
        return 0

    def evaluate_student(self, epoch, loader, log_prefix='test'):
        self.model_encoder.eval()
        self.model_studentHead.eval()

        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)

            data = data.squeeze().to(self.dev)
            # get student output of instance
            with torch.no_grad():
                feat = self.model_encoder(data)
                patch_prediction = self.model_studentHead(feat)
                patch_prediction = torch.softmax(patch_prediction, dim=1)

            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]

        instance_auc_ByStudent = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred.reshape(-1))
        self.writer.add_scalar('{}_instance_AUC'.format(log_prefix), instance_auc_ByStudent, epoch)
        return 0

    def eval(self, path):
        epoch = self.load(path)
        self.evaluate_student(epoch)
        return

    def load(self, path):
        state_dicts = torch.load(path)
        epoch = state_dicts['Epoch']
        self.model_encoder.load_state_dict(state_dicts['model_encoder'])
        # self.model_studentHead.load_state_dict(state_dicts['model_studentHead'])
        print("Model loaded")
        return epoch

    def fine_tune(self, path):
        if path != "":
            print("Loading model from {}".format(path))
            self.load(path)
        else:
            print("No model loaded, training from scratch")
        self.optimize()
        return


class Instance_Classifier_Head_simple(nn.Module):
    def __init__(self, num_classes, init=True, input_feat_dim=512):
        super(Instance_Classifier_Head_simple, self).__init__()
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(input_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        if init:
            self._initialize_weights()

    def forward(self, x):
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=10000, type=int, help='multiply LR by 0.5 every (default: 150 epochs)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64', choices=['f64', 'f32'], type=str, help='SK-algo dtype (default: f64)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=2, type=int, help='number workers (default: 6)')
    parser.add_argument('--comment', default='PretrainedEncoder', type=str, help='name for tensorboardX')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=10, type=int, help='random seed')

    parser.add_argument('--pretrain_path', default='', type=str, help='name for tensorboardX')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_%s" % args.comment.replace('/', '_') + \
           "_Seed{}_Bs{}_lr{}".format(
               args.seed, args.batch_size, args.lr,
           )
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device
    util.setup_runtime(seed=args.seed, cuda_dev_id=list(np.unique(args.modeldevice + args.device)))

    print(name, flush=True)

    writer = SummaryWriter('./runs_pathology_withGTInstLabel/%s'%name)
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    # Setup model
    model_encoder = PretrainedResNet18_Encoder().to('cuda:0')
    # model_studentHead = student_head(num_classes=[2], input_feat_dim=512).to('cuda:0')
    model_studentHead = Instance_Classifier_Head_simple(num_classes=2, input_feat_dim=512).to('cuda:0')

    optimizer_encoder = torch.optim.SGD(model_encoder.parameters(), lr=args.lr)
    optimizer_studentHead = torch.optim.SGD(model_studentHead.parameters(), lr=args.lr)

    # Setup loaders
    ds_train, ds_test, ds_test_withInstLabel = gather_align_EndoImg_Center_forPathology(
        root_dir='',
        pred_dir='',
        labeled_dir=''
    )
    ds_External_test_withInstLabel = gather_external_Pathology(root_dir='')
    ds_combined_test_withInstLabel = ds_test_withInstLabel + ds_External_test_withInstLabel
    ds_pathology_train, ds_pathology_test = split_GT_pathology_alongPatient(ds_External_test_withInstLabel)

    train_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),

        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4738405, 0.30310306, 0.20231445],
                             std=[0.2768913, 0.2088029, 0.16106644])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4738405, 0.30310306, 0.20231445],
                             std=[0.2768913, 0.2088029, 0.16106644])
    ])

    train_ds_return_instance = Endo_img_MIL_all_center_withPathologyLabel(
        ds=ds_pathology_train, transform=train_transform, num_instances_per_bag=1, task_classes=2, certainty_threshold=0.5)
    val_ds_return_instance = Endo_img_MIL_all_center_withPathologyLabel(
        ds=ds_pathology_test, transform=val_transform, num_instances_per_bag=1, task_classes=2, certainty_threshold=0.5)

    train_loader_instance = torch.utils.data.DataLoader(train_ds_return_instance, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader_instance = torch.utils.data.DataLoader(val_ds_return_instance, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    print("[Data] {} training samples".format(len(train_loader_instance.dataset)))
    print("[Data] {} evaluating samples".format(len(val_loader_instance.dataset)))

    # Setup optimizer
    o = Optimizer(model_encoder=model_encoder,
                  model_studentHead=model_studentHead,

                  optimizer_encoder=optimizer_encoder,
                  optimizer_studentHead=optimizer_studentHead,

                  train_instanceloader=train_loader_instance,
                  test_instanceloader=val_loader_instance,

                  writer=writer, num_epoch=args.epochs,
                  dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Optimize
    o.fine_tune(path=args.pretrain_path)

