import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from kan import KAN
from resnet18 import Resnet

class CTKAN(nn.Module):
    def __init__(self, num_classes=10, flrn = 5):
        super(CTKAN, self).__init__()
        
        # 使用resnet
        self.resnet = Resnet(num_classes=num_classes, freeze_until_layer=flrn)
        
        # 获取 ResNet 的输出特征维度
        self.feature_dim = 512  
        
        #  LSTM 处理全局特征
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # KAN 结构（此处简化为全连接层）
        self.kan = KAN([128, 64, num_classes])

    def forward(self, x):
        # print(f"x.shape: {x.shape}")
        batch_size = x.size(0)
        
        x =  self.resnet.resnet(x)  # 这里只用Resnet类里的resnet提取特征，就不会使用全连接层
        # print(f"After CNN, x.shape: {x.shape}")
        
        # x = x.view(batch_size, -1, 128*4*4) 
        x = x.view(batch_size, 1, self.feature_dim)
        # print(f"After View, x.shape: {x.shape}")
        
        x, _ = self.lstm(x)  # LSTM 处理时序特征
        # print(f"After LSTM, x.shape: {x.shape}")
        
        x = x[:, -1, :]  # 取最后一个时间步的输出
        # print(f"After Attain Last Step from LSTM, x.shape: {x.shape}")
        
        x = self.kan(x)  # KAN 进行最终分类
        # print(f"After kan, x.shape: {x.shape}")
        return x