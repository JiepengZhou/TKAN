import torch
import torch.nn as nn
import torchvision.models as models
from MyKan import KAN 

class CTKAN(nn.Module):
    def __init__(self, num_classes=10, flrn=5):
        super(CTKAN, self).__init__()
        
        # 使用 ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 获取 ResNet 的输出特征维度
        self.feature_dim = self.resnet.fc.in_features
        
        # 去掉 ResNet 的分类头
        self.resnet.fc = nn.Identity()

        # 冻结部分层（前 flrn 层）
        children = list(self.resnet.children())[:-2]  # 除去分类头和全连接层部分
        for child in children[:flrn]:
            for param in child.parameters():
                param.requires_grad = False

        # LSTM 处理全局特征
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # KAN 结构（此处简化为全连接层）
        self.kan = KAN([128, 64, num_classes])

    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.resnet(x)  # ResNet 提取特征
        
        x = x.view(batch_size, 1, self.feature_dim)  # 调整形状以匹配 LSTM 输入
        
        x, _ = self.lstm(x)  # LSTM 处理时序特征
        
        x = x[:, -1, :]  # 取最后一个时间步的输出
        
        x = self.kan(x)  # KAN 进行最终分类
        
        return x