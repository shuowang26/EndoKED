import argparse
import datetime
import logging
import os
import random
import sys

sys.path.append("/data/PROJECTS/Endo_GPT/EndoGPT_WSSS/")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import dataset_loader_zhongshan_ALL as voc
from datasets.dataset_loader_zhongshan_ALL import load_img_info_from_csv_public_2w, load_test_img_mask,load_img_info_from_pub_train
from model.losses import get_masked_ptc_loss, get_seg_loss, get_seg_loss_update, CTCLoss_neg, DenseEnergyLoss, get_energy_loss, DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss
from model.model_seg_neg import network
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer,imutils2
from utils.camutils_zhongshan_all import cam_to_label, cam_to_roi_mask2, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2, crop_from_roi_neg
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
torch.hub.set_dir("./pretrained")
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser()

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--data_folder", default='../VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=2, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=112, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_voc_wseg", type=str, help="work_dir_voc_wseg")

parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 0.7), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=80000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.75, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.45, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 0.75, 1.25, 1.5), help="multi_scales for cam")

parser.add_argument("--w_ptc", default=0.2, type=float, help="w_ptc")
parser.add_argument("--w_ctc", default=0, type=float, help="w_ctc")
parser.add_argument("--w_seg", default=0.1, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--w_dice", default=0.1, type=float, help="w_dice")

parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.99, type=float, help="temp")
parser.add_argument("--aux_layer", default=-2, type=int, help="aux_layer")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", default=True, help="fix random seed")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=8, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def validate(model=None, data_loader=None, args=None, n_iter=1,writer=None):

    preds, gts, cams, cams_aux = [], [], [], []
    dice_all = 0.0
    hd95_all = 0.0
    idx = 0
    visual_idx = torch.randint(0,len(data_loader),(1,))
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            idx += 1
            img_name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs  = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls, segs, _, _ = model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy(), cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            _,cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            _,cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            # for pred, gt in zip (preds,gts):
            dice, hd95 = evaluate.cal_dice(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16),labels.cpu().numpy().astype(np.int16))
            dice_all += dice
            hd95_all += hd95

            if idx == visual_idx:
                 # img visual
                preds_ = torch.argmax(resized_segs,dim=1,).cpu().numpy().astype(np.int16)
                pseudo_gts = cam_label.cpu().numpy().astype(np.int16)
                pseudo_gts_aux = cam_label_aux.cpu().numpy().astype(np.int16)
                true_gts = labels.cpu().numpy().astype(np.int16)
                grid_imgs, grid_cam = imutils2.tensorboard_image_solo(imgs=inputs.clone(), cam=resized_cam)
                _, grid_cam_aux = imutils2.tensorboard_image_solo(imgs=inputs.clone(), cam=resized_cam_aux)

                grid_preds = imutils2.tensorboard_label_solo(labels=preds_)
                grid_pseudo_gt = imutils2.tensorboard_label_solo(labels=pseudo_gts)
                grid_true_gt = imutils2.tensorboard_label_solo(labels=true_gts)
                grid_pseudo_gt_aux = imutils2.tensorboard_label_solo(labels=pseudo_gts_aux)
                writer.add_image("val/images", grid_imgs, global_step=n_iter)
                writer.add_image("val/gts", grid_true_gt, global_step=n_iter)
                writer.add_image("val/preds", grid_preds, global_step=n_iter)
                writer.add_image("val/pseudo_pseudo_gts", grid_pseudo_gt, global_step=n_iter)
                writer.add_image("val/pseudo_pseudo_gts_aux", grid_pseudo_gt_aux, global_step=n_iter)

                #writer.add_image("train/pseudo_irn_gts", grid_irn_gt, global_step=n_iter)
                # writer.add_image("val_cam/valid_cams", grid_cam, global_step=n_iter)
                # writer.add_image("val_cam/aux_cams", grid_cam_aux, global_step=n_iter)

            # valid_label = torch.nonzero(cls_label[0])[:, 0]
            # out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})
    
    dice_mean = dice_all / idx
    hd95_mean = hd95_all / idx
    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds,num_classes=args.num_classes)
    cam_score = evaluate.scores(gts, cams,num_classes=args.num_classes)
    cam_aux_score = evaluate.scores(gts, cams_aux,num_classes=args.num_classes)

    writer.add_scalars('val/loss', { "cls_loss": cls_score.item()}, global_step=n_iter)
    writer.add_scalars('val/dice', { "dice": dice_mean}, global_step=n_iter)
    writer.add_scalars('val/hd95', { "hd95": hd95_mean}, global_step=n_iter)

    model.train()

    tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"], cat_list=["Background","Endo"])

    return cls_score, tab_results, dice_mean, hd95_mean

def train(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    root_dir_train = '/data/Datasets/MedicalDatasets/息肉数据集/endo_gpt_zhongshan_all/network_MIL_prediction_results/train/'
    root_dir_test = "/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/CVC-ClinicDB_datasets/CVC-ClinicDB_datasets/CVC-ClinicDB_PNG_datasets/"
    public_2w_path = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/public_2w/V1/'
    # root_dir_test = '/data/PROJECTS/Endo_GPT/Datasets/test'
    img_path_list_train, label_train = load_img_info_from_csv_public_2w(root_dir_train,public_2w_path)


    pub_train_dir = '/data/Datasets/MedicalDatasets/息肉数据集/息肉公开数据集/dataset/TrainDataset/'
    img_path_list_train, label_train =load_img_info_from_pub_train(root_dir_train,pub_train_dir)

    
    img_path_list_test, mask_test_path = load_test_img_mask(root_dir_test)
    # img_path_list_test, label_test = load_img_label_info_from_csv(root_dir_test)
    
    train_dataset = voc.Endo_img_WSSS(
        img_path_list_train,
        label_train,
        type='train',
        resize_range=[512, 640],
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=255,
        num_classes=args.num_classes,
        aug=True,
    )

    val_dataset = voc.Endo_img_WSSS(
        img_path_list_test,
        mask_test_path,
        type='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        #shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        init_momentum=args.momentum,
        aux_layer=args.aux_layer
    )
    param_groups = model.get_param_groups()
    model.to(device)
    if args.local_rank==0:
        writer = SummaryWriter(args.tb_dir)
    # cfg.optimizer.learning_rate *= 2
    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)

    logging.info('\nOptimizer: \n%s' % optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    ncrops = 10
    CTC_loss = CTCLoss_neg(ncrops=ncrops, temp=args.temp).cuda()
    DICE_loss = DiceLoss(args.num_classes).to(device)


    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()

    for n_iter in range(args.max_iters):

        try:
            inputs, cls_label, img_box, crops = next(train_loader_iter)

        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            inputs, cls_label, img_box, crops = next(train_loader_iter)
        # b,1,2 -> b,4,1,1
        # cls_label = cls_label.squeeze(1)
        # inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(device, non_blocking=True)
        inputs = inputs.type(torch.cuda.FloatTensor)

        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        # get local crops from uncertain regions

        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)
        roi_mask = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=args.low_thre, hig_thre=args.high_thre)

        local_crops, flags = crop_from_roi_neg(images=crops[2], roi_mask=roi_mask, crop_num=ncrops-2, crop_size=args.local_crop_size)
        roi_crops = crops[:2] + local_crops

        cls, segs, fmap, cls_aux, out_t, out_s = model(inputs, crops=roi_crops, n_iter=n_iter)

        # cls loss & aux cls loss
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        # ctc_loss
        ctc_loss = CTC_loss(out_s, out_t, flags)

        # ptc loss
        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)
        # ptc_loss = get_ptc_loss(fmap, low_fmap) 

        # seg_loss & reg_loss
        valid_cam_aux,_ = cam_to_label(cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        refined_pseudo_label_aux = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam_aux, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        if n_iter >= 12000:
            supervison_mask = refined_pseudo_label
        else:
            supervison_mask = refined_pseudo_label_aux
        segs = F.interpolate(segs, size=supervison_mask.shape[1:], mode='bilinear', align_corners=False)
        ce_loss = CrossEntropyLoss(ignore_index=args.ignore_index, reduction='mean')
        # seg_loss = get_seg_loss_update(segs, supervison_mask.type(torch.long), ignore_index=args.ignore_index)
        seg_loss = ce_loss(segs, supervison_mask.type(torch.long))
        # SEG_Loss = CrossEntropyLoss(ignore_index=args.ignore_index).to(device)
        # seg_loss = SEG_Loss(segs, refined_pseudo_label_aux.type(torch.long))

        # reg_loss = get_energy_loss(img=inputs, logit=segs, label=supervison_mask, img_box=img_box, loss_layer=loss_layer)

        dice_loss = DICE_loss(segs, supervison_mask.type(torch.long), softmax=True)

        # warmup
        if n_iter <= 1000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss + 0.0 * seg_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss + args.w_seg * seg_loss + args.w_dice * dice_loss

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'ctc_loss': ctc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'dice_loss': dice_loss.item(),
            'cls_score': cls_score.item(),
        })

        optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        # 100 record tensorboard
        if (n_iter + 1) % 20 == 0:
            preds = torch.argmax(segs,dim=1,).cpu().numpy().astype(np.int16)
            refined_gts = refined_pseudo_label.cpu().numpy().astype(np.int16)
            refined_gts_aux = refined_pseudo_label_aux.cpu().numpy().astype(np.int16)
            grid_imgs, grid_cam = imutils2.tensorboard_image(imgs=inputs.clone(), cam=valid_cam,nrow=4)
            _, grid_cam_aux = imutils2.tensorboard_image(imgs=inputs.clone(), cam=valid_cam_aux,nrow=4)

            grid_preds = imutils2.tensorboard_label(labels=preds,nrow=4)
            grid_refined_gt = imutils2.tensorboard_label(labels=refined_gts,nrow=4)
            grid_refined_gt_aux = imutils2.tensorboard_label(labels=refined_gts_aux,nrow=4)

            writer.add_image("train/images", grid_imgs, global_step=n_iter)
            writer.add_image("train/preds", grid_preds, global_step=n_iter)
            writer.add_image("train/pseudo_pseudo_gts", grid_refined_gt, global_step=n_iter)
            writer.add_image("train/pseudo_pseudo_gts_aux", grid_refined_gt_aux, global_step=n_iter)

            #writer.add_image("train/pseudo_irn_gts", grid_irn_gt, global_step=n_iter)
            writer.add_image("cam/valid_cams", grid_cam, global_step=n_iter)
            writer.add_image("cam/aux_cams", grid_cam_aux, global_step=n_iter)

        writer.add_scalars('train/loss', {"seg_loss": seg_loss.item(),"dice_loss": dice_loss.item(), "cls_loss": cls_loss.item(),"total_loss": loss.item(),"ptc_loss": ptc_loss.item(),"ctc_loss": ctc_loss.item(),}, global_step=n_iter)

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, ctc_loss: %.4f, seg_loss: %.4f,dice_loss: %.4f..." % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('ptc_loss'), avg_meter.pop('ctc_loss'), avg_meter.pop('seg_loss'), avg_meter.pop('dice_loss')))

        if (n_iter + 1) % args.eval_iters == 0:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
            val_cls_score, tab_results, dice_mean, hd95_mean = validate(model=model, data_loader=val_loader,n_iter=n_iter, args=args,writer=writer)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("val dice: %.6f" % (dice_mean))
                logging.info("val hd95 %.6f" % (hd95_mean))
                logging.info("\n"+tab_results)

    return True


if __name__ == "__main__":

    args = parser.parse_args()

    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")
    args.tb_dir = os.path.join(args.work_dir, "tensorboard_log")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)
