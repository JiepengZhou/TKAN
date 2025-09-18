import os
import torch.nn as nn
import torchvision.models as models
from kan import KAN

class CTKAN(nn.Module):
    def __init__(self, num_classes=10, flrn=5):
        super(CTKAN, self).__init__()

        # 使用 ConvNeXt
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # 获取 ConvNeXt 的输出特征维度
        self.feature_dim = self.convnext.classifier[2].in_features
        
        # 去掉 ConvNeXt 的分类头（只保留特征提取部分）
        self.convnext.classifier = nn.Identity()

        # 冻结前 flrn 层
        children = list(self.convnext.features.children())
        for child in children[:flrn]:
            for param in child.parameters():
                param.requires_grad = False
        
        # LSTM 处理全局特征
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # KAN 结构（此处简化为全连接层）
        self.kan = KAN([128, 64, num_classes])

    def forward(self, x):
        batch_size = x.size(0)

        # ConvNeXt 特征提取
        x = self.convnext(x)
        
        # 调整输入形状以匹配 LSTM 的输入
        x = x.view(batch_size, 1, self.feature_dim)
        
        # LSTM 处理时序特征
        x, _ = self.lstm(x)
        
        # 取 LSTM 最后一个时间步的输出
        x = x[:, -1, :]
        
        # KAN 进行最终分类
        x = self.kan(x)
        
        return x