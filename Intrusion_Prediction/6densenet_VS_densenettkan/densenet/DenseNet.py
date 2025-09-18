import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义 DenseNet 模型
class DenseNet_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet_Model, self).__init__()
        
        # 加载预训练的 DenseNet 模型
        self.densenet = models.densenet121(weights=models.densenet.DenseNet121_Weights.IMAGENET1K_V1)
        
        # 修改分类头（全连接层）
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)
