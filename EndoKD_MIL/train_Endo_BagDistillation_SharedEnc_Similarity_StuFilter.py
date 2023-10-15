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
from Datasets_loader.dataset_Endo import gather_align_EndoImg, Endo_img_MIL
import datetime
import utliz
import util
import random
from tqdm import tqdm
import copy
import pandas as pd


class Optimizer:
    def __init__(self, model_encoder, model_teacherHead, model_studentHead,
                 optimizer_encoder, optimizer_teacherHead, optimizer_studentHead,
                 train_bagloader, train_instanceloader, test_bagloader, test_instanceloader,
                 writer=None, num_epoch=100,
                 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 PLPostProcessMethod='NegGuide', StuFilterType='ReplaceAS', smoothE=100,
                 stu_loss_weight_neg=0.1, stuOptPeriod=1,
                 teacher_loss_weight=[1.0, 1.0],
                 teacher_pseudo_label_merge_weight=[0.5, 0.5]):
        self.model_encoder = model_encoder
        self.model_teacherHead = model_teacherHead
        self.model_studentHead = model_studentHead
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_teacherHead = optimizer_teacherHead
        self.optimizer_studentHead = optimizer_studentHead
        self.train_bagloader = train_bagloader
        self.train_instanceloader = train_instanceloader
        self.test_bagloader = test_bagloader
        self.test_instanceloader = test_instanceloader
        self.writer = writer
        self.num_epoch = num_epoch
        self.dev = dev
        self.log_period = 10
        self.PLPostProcessMethod = PLPostProcessMethod
        self.StuFilterType = StuFilterType
        self.smoothE = smoothE
        self.stu_loss_weight_neg = stu_loss_weight_neg
        self.stuOptPeriod = stuOptPeriod
        self.num_teacher = len(model_teacherHead)
        self.teacher_loss_weight = teacher_loss_weight
        self.teacher_pseudo_label_merge_weight = teacher_pseudo_label_merge_weight

    def optimize(self):
        self.Bank_all_Bags_label = None
        self.Bank_all_instances_pred_byTeacher = None
        self.Bank_all_instances_feat_byTeacher = None
        self.Bank_all_instances_pred_processed = None
        self.Bank_all_instances_pred_byStudent = None

        for epoch in range(self.num_epoch):
            self.optimize_teacher(epoch)
            self.evaluate_teacher(epoch)
            if epoch % self.stuOptPeriod == 0:
                self.optimize_student(epoch)
                self.evaluate_student(epoch)
        return 0

    def optimize_teacher(self, epoch):
        self.model_encoder.train()
        for model_teacherHead_i in self.model_teacherHead:
            model_teacherHead_i.train()
        self.model_studentHead.eval()
        ## optimize teacher with bag-dataloader
        # 1. change loader to bag-loader
        loader = self.train_bagloader
        # 2. optimize
        patch_label_gt = []
        bag_label_gt = []
        patch_label_pred = [[] for i in range(self.num_teacher)]
        bag_label_pred = [[] for i in range(self.num_teacher)]
        patch_corresponding_bag_label = []
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Teacher training')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            feat = self.model_encoder(data.squeeze(0))

            if epoch > self.smoothE and label[1] == 1:
                pass
            else:
                loss_teacher = 0
                for i in range(self.num_teacher):
                    bag_prediction_teacher_i, instance_attn_score_teacher_i = self.model_teacherHead[i](feat)
                    bag_prediction_teacher_i = torch.softmax(bag_prediction_teacher_i, 1)
                    loss_teacher_i = self.model_teacherHead[i].get_loss(bag_prediction_teacher_i, instance_attn_score_teacher_i, label[1])
                    loss_teacher = loss_teacher + self.teacher_loss_weight[i] * loss_teacher_i

                    patch_label_pred[i].append(instance_attn_score_teacher_i.detach().squeeze(0))
                    bag_label_pred[i].append(bag_prediction_teacher_i.detach()[0, 1])

                self.optimizer_encoder.zero_grad()
                for optimizer_teacherHead_i in self.optimizer_teacherHead:
                    optimizer_teacherHead_i.zero_grad()
                loss_teacher.backward()
                self.optimizer_encoder.step()
                for optimizer_teacherHead_i in self.optimizer_teacherHead:
                    optimizer_teacherHead_i.step()

            patch_label_gt.append(label[0].squeeze(0))
            bag_label_gt.append(label[1])
            patch_corresponding_bag_label.append(torch.ones([data.shape[1]]).to(self.dev)*label[1])
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Teacher', loss_teacher.item(), niter)

        patch_label_pred = [torch.cat(i) for i in patch_label_pred]
        bag_label_pred = [torch.tensor(i) for i in bag_label_pred]
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_gt = torch.cat(bag_label_gt)
        patch_corresponding_bag_label = torch.cat(patch_corresponding_bag_label)

        self.estimated_AttnScore_norm_para_min = [i.min() for i in patch_label_pred]
        self.estimated_AttnScore_norm_para_max = [i.max() for i in patch_label_pred]
        for i in range(self.num_teacher):
            patch_label_pred_normed = self.norm_AttnScore2Prob(patch_label_pred[i], idx_teacher=i)
            # instance_auc_ByTeacher = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred_normed.reshape(-1))
            bag_auc_ByTeacher = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_pred[i].reshape(-1))
            # self.writer.add_scalar('train_instance_AUC_byTeacher{}'.format(i), instance_auc_ByTeacher, epoch)
            self.writer.add_scalar('train_bag_AUC_byTeacher{}'.format(i), bag_auc_ByTeacher, epoch)
        # print("Epoch:{} train_bag_AUC_byTeacher:{}".format(epoch, bag_auc_ByTeacher))

        return 0

    def norm_AttnScore2Prob(self, attn_score, idx_teacher):
        prob = (attn_score - self.estimated_AttnScore_norm_para_min[idx_teacher]) / \
               (self.estimated_AttnScore_norm_para_max[idx_teacher] - self.estimated_AttnScore_norm_para_min[idx_teacher])
        return prob

    def optimize_student(self, epoch):
        for model_teacherHead_i in self.model_teacherHead:
            model_teacherHead_i.train()
        self.model_encoder.train()
        self.model_studentHead.train()
        ## optimize teacher with instance-dataloader
        # 1. change loader to instance-loader
        loader = self.train_instanceloader
        # 2. optimize
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        bag_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student training')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            # get teacher output of instance
            feat = self.model_encoder(data)
            pseudo_instance_label = torch.zeros_like(label[0])
            with torch.no_grad():
                for i in range(self.num_teacher):
                    _, instance_attn_score = self.model_teacherHead[i](feat)
                    pseudo_instance_label = pseudo_instance_label + self.teacher_pseudo_label_merge_weight[i] * \
                                            self.norm_AttnScore2Prob(instance_attn_score, idx_teacher=i).clamp(min=1e-5, max=1-1e-5).squeeze(0)
            # set true negative patch label to [1, 0]
            pseudo_instance_label[label[1] == 0] = 0
            # # DEBUG: Assign GT patch label
            # pseudo_instance_label = label[0]
            # get student output of instance
            patch_prediction = self.model_studentHead(feat)
            patch_prediction = torch.softmax(patch_prediction, dim=1)

            # cal loss
            loss_student = -1. * torch.mean(self.stu_loss_weight_neg * (1-pseudo_instance_label) * torch.log(patch_prediction[:, 0] + 1e-5) +
                                            (1-self.stu_loss_weight_neg) * pseudo_instance_label * torch.log(patch_prediction[:, 1] + 1e-5))
            self.optimizer_encoder.zero_grad()
            self.optimizer_studentHead.zero_grad()
            loss_student.backward()
            self.optimizer_encoder.step()
            self.optimizer_studentHead.step()

            patch_corresponding_slide_idx[selected, 0] = label[2]
            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]
            bag_label_gt[selected, 0] = label[1]
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Student', loss_student.item(), niter)


        # cal bag-level auc
        bag_label_gt_coarse = []
        bag_label_prediction = []
        available_bag_idx = patch_corresponding_slide_idx.unique()
        for bag_idx_i in available_bag_idx:
            idx_same_bag_i = torch.where(patch_corresponding_slide_idx == bag_idx_i)
            if bag_label_gt[idx_same_bag_i].max() != bag_label_gt[idx_same_bag_i].max():
                raise
            bag_label_gt_coarse.append(bag_label_gt[idx_same_bag_i].max())
            bag_label_prediction.append(patch_label_pred[idx_same_bag_i].max())
        bag_label_gt_coarse = torch.tensor(bag_label_gt_coarse)
        bag_label_prediction = torch.tensor(bag_label_prediction)
        bag_auc_ByStudent = utliz.cal_auc(bag_label_gt_coarse.reshape(-1), bag_label_prediction.reshape(-1))
        self.writer.add_scalar('train_bag_AUC_byStudent', bag_auc_ByStudent, epoch)

        return 0

    def evaluate_teacher(self, epoch):
        self.model_encoder.eval()
        for model_teacherHead_i in self.model_teacherHead:
            model_teacherHead_i.eval()
        self.model_studentHead.eval()
        ## optimize teacher with bag-dataloader
        # 1. change loader to bag-loader
        loader = self.test_bagloader
        # 2. optimize
        patch_label_gt = []
        bag_label_gt = []
        patch_label_pred = [[] for i in range(self.num_teacher)]
        bag_label_prediction_withAttnScore = [[] for i in range(self.num_teacher)]
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Teacher evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            with torch.no_grad():
                feat = self.model_encoder(data.squeeze(0))
                for i in range(self.num_teacher):
                    bag_prediction_withAttnScore_i, instance_attn_score = self.model_teacherHead[i](feat)
                    bag_prediction_withAttnScore_i = torch.softmax(bag_prediction_withAttnScore_i, 1)

                    patch_label_pred[i].append(instance_attn_score.detach().squeeze(0))
                    bag_label_prediction_withAttnScore[i].append(bag_prediction_withAttnScore_i.detach()[0, 1])
            patch_label_gt.append(label[0].squeeze(0))
            bag_label_gt.append(label[1])

        patch_label_pred = [torch.cat(i) for i in patch_label_pred]
        bag_label_prediction_withAttnScore = [torch.tensor(i) for i in bag_label_prediction_withAttnScore]
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_gt = torch.cat(bag_label_gt)

        for i in range(self.num_teacher):
            patch_label_pred_normed = (patch_label_pred[i] - patch_label_pred[i].min()) / (patch_label_pred[i].max() - patch_label_pred[i].min())
            # instance_auc_ByTeacher = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred_normed.reshape(-1))
            bag_auc_ByTeacher_withAttnScore = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_prediction_withAttnScore[i].reshape(-1))
            # self.writer.add_scalar('test_instance_AUC_byTeacher{}'.format(i), instance_auc_ByTeacher, epoch)
            self.writer.add_scalar('test_bag_AUC_byTeacher{}'.format(i), bag_auc_ByTeacher_withAttnScore, epoch)
        return 0

    def evaluate_student(self, epoch):
        self.model_encoder.eval()
        self.model_studentHead.eval()
        ## optimize teacher with instance-dataloader
        # 1. change loader to instance-loader
        loader = self.test_instanceloader
        # 2. optimize
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        bag_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            # get student output of instance
            with torch.no_grad():
                feat = self.model_encoder(data)
                patch_prediction = self.model_studentHead(feat)
                patch_prediction = torch.softmax(patch_prediction, dim=1)

            patch_corresponding_slide_idx[selected, 0] = label[2]
            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]
            bag_label_gt[selected, 0] = label[1]


        # cal bag-level auc
        bag_label_gt_coarse = []
        bag_label_prediction = []
        available_bag_idx = patch_corresponding_slide_idx.unique()
        for bag_idx_i in available_bag_idx:
            idx_same_bag_i = torch.where(patch_corresponding_slide_idx == bag_idx_i)
            if bag_label_gt[idx_same_bag_i].max() != bag_label_gt[idx_same_bag_i].max():
                raise
            bag_label_gt_coarse.append(bag_label_gt[idx_same_bag_i].max())
            bag_label_prediction.append(patch_label_pred[idx_same_bag_i].max())

        bag_label_gt_coarse = torch.tensor(bag_label_gt_coarse)
        bag_label_prediction = torch.tensor(bag_label_prediction)
        bag_auc_ByStudent = utliz.cal_auc(bag_label_gt_coarse.reshape(-1), bag_label_prediction.reshape(-1))
        self.writer.add_scalar('test_bag_AUC_byStudent', bag_auc_ByStudent, epoch)


        return 0


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
    parser.add_argument('--batch_size', default=128, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=1000, type=int, help='multiply LR by 0.5 every (default: 150 epochs)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64', choices=['f64', 'f32'], type=str, help='SK-algo dtype (default: f64)')

    # SK algo
    parser.add_argument('--nopts', default=100, type=int, help='number of pseudo-opts (default: 100)')
    parser.add_argument('--augs', default=3, type=int, help='augmentation level (default: 3)')
    parser.add_argument('--lamb', default=25, type=int, help='for pseudoopt: lambda (default:25) ')

    # architecture
    # parser.add_argument('--arch', default='alexnet_MNIST', type=str, help='alexnet or resnet (default: alexnet)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=0, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='DEBUG_MultiTeacher_newPHM', type=str, help='name for tensorboardX')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=10, type=int, help='random seed')

    parser.add_argument('--dataset_downsampling', default=0.1, type=float, help='sampling the dataset for Debug')

    parser.add_argument('--PLPostProcessMethod', default='NegGuide', type=str,
                        help='Post-processing method of Attention Scores to build Pseudo Lables',
                        choices=['NegGuide', 'NegGuide_TopK', 'NegGuide_Similarity'])
    parser.add_argument('--StuFilterType', default='PseudoBag_85_15_2', type=str,
                        help='Type of using Student Prediction to imporve Teacher '
                             '[ReplaceAS, FilterNegInstance_Top95, FilterNegInstance_ThreProb95, PseudoBag_88_20]')
    parser.add_argument('--smoothE', default=0, type=int, help='num of epoch to apply StuFilter')
    parser.add_argument('--stu_loss_weight_neg', default=1.0, type=float, help='weight of neg instances in stu training')
    parser.add_argument('--stuOptPeriod', default=1, type=int, help='period of stu optimization')
    parser.add_argument('--TeacherLossWeight', nargs='+', type=float, help='weight of multiple teacher, like: 1.0 1.0', required=True)
    parser.add_argument('--PLMergeWeight', nargs='+', type=float, help='weight of merge teachers pseudo label, like: 0.5 0.5', required=True)
    return parser.parse_args()


class map_abmil(nn.Module):
    def __init__(self, model):
        super(map_abmil, self).__init__()
        self.model = model

    def forward(self, x):
        bag_prediction, _, _, instance_attn_score = self.model(x, returnBeforeSoftMaxA=True, scores_replaceAS=None)
        if len(instance_attn_score.shape)==1:
            instance_attn_score = instance_attn_score.unsqueeze(0)
        return bag_prediction, instance_attn_score

    def get_loss(self, output_bag, output_inst, bag_label):
        # output shape: 1
        # label shape: 1
        output_bag = output_bag[0, 1]
        bag_label = bag_label.squeeze()
        loss = -1. * (bag_label * torch.log(output_bag+1e-5) + (1. - bag_label) * torch.log(1. - output_bag+1e-5))
        return loss


class map_dsmil(nn.Module):
    def __init__(self, model):
        super(map_dsmil, self).__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        instance_attn_score, bag_prediction, _, _ = self.model(x)
        instance_attn_score = torch.softmax(instance_attn_score, dim=1)
        instance_attn_score = instance_attn_score[:, 1].unsqueeze(0)
        return bag_prediction, instance_attn_score

    def get_loss(self, output_bag, output_inst, bag_label):
        # output shape: 1
        # label shape: 1
        max_id = torch.argmax(output_inst.squeeze(0))
        bag_pred_byMax = output_inst.squeeze(0)[max_id]
        bag_loss = self.criterion(output_bag, bag_label)

        bag_label = bag_label.squeeze()
        bag_loss_byMax = -1. * (bag_label * torch.log(bag_pred_byMax+1e-5) + (1. - bag_label) * torch.log(1. - bag_pred_byMax+1e-5))
        loss = 0.5 * bag_loss + 0.5 * bag_loss_byMax
        return loss


if __name__ == "__main__":
    args = get_parser()

    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_%s" % args.comment.replace('/', '_') + \
           "_Seed{}_Bs{}_lr{}_Downsample{}_PLPostProcessBy{}_StuFilterType{}_smoothE{}_weightN{}_StuOptP{}_TeacherLossW{}_MergeW{}".format(
               args.seed, args.batch_size, args.lr, args.dataset_downsampling,
               args.PLPostProcessMethod, args.StuFilterType, args.smoothE, args.stu_loss_weight_neg, args.stuOptPeriod,
               str(args.TeacherLossWeight), str(args.PLMergeWeight),
           )
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device
    util.setup_runtime(seed=42, cuda_dev_id=list(np.unique(args.modeldevice + args.device)))

    print(name, flush=True)

    writer = SummaryWriter('./runs_Endo/%s'%name)
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    # Setup model
    model_encoder = PretrainedResNet18_Encoder().to('cuda:0')
    model_studentHead = student_head(input_feat_dim=512).to('cuda:0')
    model_teacherHead = map_abmil(teacher_Attention_head(input_feat_dim=512).to('cuda:0'))
    model_teacherHead_2 = map_dsmil(teacher_DSMIL_head(input_feat_dim=512).to('cuda:0'))

    optimizer_encoder = torch.optim.SGD(model_encoder.parameters(), lr=args.lr)
    optimizer_teacherHead = torch.optim.SGD(model_teacherHead.parameters(), lr=args.lr)
    optimizer_teacherHead_2 = torch.optim.SGD(model_teacherHead_2.parameters(), lr=args.lr)
    optimizer_studentHead = torch.optim.SGD(model_studentHead.parameters(), lr=args.lr)

    # Setup loaders
    ds_train, ds_test = gather_align_EndoImg()

    train_ds_return_instance = Endo_img_MIL(ds=ds_train, downsample=args.dataset_downsampling, transform=None, preload=False, return_bag=False)
    train_ds_return_bag = copy.deepcopy(train_ds_return_instance)
    train_ds_return_bag.return_bag = True
    val_ds_return_instance = Endo_img_MIL(ds=ds_test, downsample=args.dataset_downsampling, transform=None, preload=False, return_bag=False)
    val_ds_return_bag = Endo_img_MIL(ds=ds_test, downsample=args.dataset_downsampling, transform=None, preload=False, return_bag=True)

    train_loader_instance = torch.utils.data.DataLoader(train_ds_return_instance, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    train_loader_bag = torch.utils.data.DataLoader(train_ds_return_bag, batch_size=1, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader_instance = torch.utils.data.DataLoader(val_ds_return_instance, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    val_loader_bag = torch.utils.data.DataLoader(val_ds_return_bag, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False)

    print("[Data] {} training samples".format(len(train_loader_instance.dataset)))
    print("[Data] {} evaluating samples".format(len(val_loader_instance.dataset)))

    if torch.cuda.device_count() > 1:
        print("Let's use", len(args.modeldevice), "GPUs for the model")
        if len(args.modeldevice) == 1:
            print('single GPU model', flush=True)
        else:
            model_encoder = nn.DataParallel(model_encoder, device_ids=list(range(len(args.modeldevice))))
            model_teacherHead = nn.DataParallel(model_teacherHead, device_ids=list(range(len(args.modeldevice))))
            model_teacherHead_2 = nn.DataParallel(model_teacherHead_2, device_ids=list(range(len(args.modeldevice))))
            optimizer_studentHead = nn.DataParallel(optimizer_studentHead, device_ids=list(range(len(args.modeldevice))))

    # Setup optimizer
    o = Optimizer(model_encoder=model_encoder,
                  model_teacherHead=[model_teacherHead, model_teacherHead_2],
                  model_studentHead=model_studentHead,
                  optimizer_encoder=optimizer_encoder,
                  optimizer_teacherHead=[optimizer_teacherHead, optimizer_teacherHead_2],
                  optimizer_studentHead=optimizer_studentHead,
                  train_bagloader=train_loader_bag, train_instanceloader=train_loader_instance,
                  test_bagloader=val_loader_bag, test_instanceloader=val_loader_instance,
                  writer=writer, num_epoch=args.epochs,
                  dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  PLPostProcessMethod=args.PLPostProcessMethod, StuFilterType=args.StuFilterType, smoothE=args.smoothE,
                  stu_loss_weight_neg=args.stu_loss_weight_neg, stuOptPeriod=args.stuOptPeriod,
                  teacher_loss_weight=args.TeacherLossWeight,
                  teacher_pseudo_label_merge_weight=args.PLMergeWeight)
    # Optimize
    o.optimize()
