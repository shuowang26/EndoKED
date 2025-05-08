import models
import torch
import cv2
import os
from PIL import Image
import numpy as np


def generate_model(opt):
    model = getattr(models, opt.model)()
    # if opt.use_gpu:
    #     model.cuda()
    #     torch.backends.cudnn.benchmark = True

    if opt.load_ckpt is not None:
        model_dict = model.state_dict()
        load_ckpt_path = os.path.join('./checkpoints/exp'+str(opt.expID)+'/', opt.load_ckpt + '.pth')
        assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
        print('Loading checkpoint......')
        checkpoint = torch.load(load_ckpt_path)
        new_dict = {k : v for k, v in checkpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

        print('Done')

    return model


def save_binary_img(x, testset, name):
    x = x.cpu().data.numpy()
    x = np.squeeze(x)  # batch_size == 1
    x *= 255
    img_save_dir = './pred/'+testset

    im = Image.fromarray(x)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    
    if im.mode == 'F':
        im = im.convert('L')
    im.save(os.path.join(img_save_dir, name[0] + '.png'))



