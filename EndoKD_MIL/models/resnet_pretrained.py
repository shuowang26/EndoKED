import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import torchvision.models as models


class PretrainedResNet18_Encoder(nn.Module):
    def __init__(self):
        super(PretrainedResNet18_Encoder, self).__init__()
        model_raw = models.resnet18(pretrained=True)
        self.pretrained_model = nn.Sequential(*list(model_raw.children())[:-1])

    def forward(self, x):
        return self.pretrained_model(x).squeeze(-1).squeeze(-1)


class PretrainedResNet50_Encoder(nn.Module):
    def __init__(self):
        super(PretrainedResNet50_Encoder, self).__init__()
        model_raw = models.resnet50(pretrained=True)
        self.pretrained_model = nn.Sequential(*list(model_raw.children())[:-1])

    def forward(self, x):
        return self.pretrained_model(x).squeeze(-1).squeeze(-1)


if __name__ == '__main__':
    model = PretrainedResNet18_Encoder()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)
    input_tensor = torch.randn(1, 3, 512, 512)
    output = model(input_tensor)
    print(output.shape)

    model = PretrainedResNet50_Encoder()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)
    input_tensor = torch.randn(1, 3, 512, 512)
    output = model(input_tensor)
    print(output.shape)
    print("END")