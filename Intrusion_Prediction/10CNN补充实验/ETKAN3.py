import torch
import torch.nn as nn
import torchvision.models as models
from MyKan import KAN 

class CTKAN(nn.Module):
    def __init__(self, num_classes=10, flrn=5):
        super(CTKAN, self).__init__()
        
        # 使用 EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # 获取 EfficientNet 的输出特征维度
        self.feature_dim = self.efficientnet.classifier[1].in_features
        
        # 移除 EfficientNet 的分类头
        self.efficientnet.classifier = nn.Identity()

        # 冻结部分层（前 flrn 层）
        children = list(self.efficientnet.features.children())
        for child in children[:flrn]:
            for param in child.parameters():
                param.requires_grad = False

        # LSTM 处理全局特征
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # KAN 结构（此处简化为全连接层）
        self.kan = KAN([128, 64, num_classes])

    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.efficientnet(x)  # EfficientNet 提取特征
        
        x = x.view(batch_size, 1, self.feature_dim)  # 调整形状以匹配 LSTM 输入
        
        x, _ = self.lstm(x)  # LSTM 处理时序特征
        
        x = x[:, -1, :]  # 取最后一个时间步的输出
        
        x = self.kan(x)  # KAN 进行最终分类
        
        return x