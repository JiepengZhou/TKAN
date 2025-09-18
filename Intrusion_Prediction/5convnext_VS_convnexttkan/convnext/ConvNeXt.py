import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义 ConvNeXt 模型
class ConvNeXt_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNeXt_Model, self).__init__()

        # 加载预训练的 ConvNeXt 模型
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # 修改分类头（全连接层）
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, num_classes)

    def forward(self, x):
        return self.convnext(x)