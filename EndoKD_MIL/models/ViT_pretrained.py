import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import torchvision.models as models
from segment_anything import sam_model_registry, build_sam, SamPredictor
from models.dinov2.hubconf import dinov2_vits14, dinov2_vitb14


class PretrainedViT_b_Encoder(nn.Module):
    def __init__(self, fix_weight=True, pooling='max'):
        super(PretrainedViT_b_Encoder, self).__init__()
        # model_raw = sam_model_registry["vit_h"](checkpoint="/root/Data2/sam_vit_h_4b8939.pth")
        model_raw = sam_model_registry["vit_b"](checkpoint="/home/xiaoyuan/Data3/sam_vit_b_01ec64.pth")
        self.pretrained_model = model_raw.image_encoder
        if pooling == 'max':
            self.pooling = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'cnn':
            self.pooling = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d(output_size=(1, 1))
            )
        else:
            raise

        if fix_weight:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        feat = self.pretrained_model(x)
        feat = self.pooling(feat).squeeze(-1).squeeze(-1)
        return feat


class DINO_PretrainedViT_s_Encoder(nn.Module):
    def __init__(self, fix_weight=False):
        super(DINO_PretrainedViT_s_Encoder, self).__init__()
        model_raw = dinov2_vits14()
        self.pretrained_model = model_raw
        if fix_weight:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        feat = self.pretrained_model(x)
        return feat


class DINO_PretrainedViT_b_Encoder(nn.Module):
    def __init__(self, fix_weight=False):
        super(DINO_PretrainedViT_b_Encoder, self).__init__()
        model_raw = dinov2_vitb14()
        self.pretrained_model = model_raw
        if fix_weight:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        feat = self.pretrained_model(x)
        return feat


if __name__ == '__main__':
    model = DINO_PretrainedViT_s_Encoder(fix_weight=True)
    input_tensor = torch.randn(2, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)

