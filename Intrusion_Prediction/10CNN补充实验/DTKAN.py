import torch
import torch.nn as nn
import torchvision.models as models
from MyKan import KAN 

class CTKAN(nn.Module):
    def __init__(self, num_classes=10, flrn=5):
        super(CTKAN, self).__init__()

        # 加载 DenseNet
        self.densenet = models.densenet121(weights=models.densenet.DenseNet121_Weights.IMAGENET1K_V1)

        # 修改第一层卷积使其支持单通道输入
        self.densenet.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # 获取 DenseNet 输出维度并移除分类头
        self.feature_dim = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()

        # 冻结前 flrn 层参数
        children = list(self.densenet.features.children())
        for child in children[:flrn]:
            for param in child.parameters():
                param.requires_grad = False

        # LSTM 用于建模全局时序（这里我们设 batch_size × seq_len=1）
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # KAN拟合高维数据
        self.kan = KAN([128, 64, num_classes])

    def forward(self, x):
        batch_size = x.size(0)

        # 提取特征
        x = self.densenet(x)  # (B, feature_dim)

        # 送入 LSTM，注意 reshape 成 (B, 1, feature_dim)
        x = x.view(batch_size, 1, self.feature_dim)
        x, _ = self.lstm(x)

        # 取 LSTM 的最后一步
        x = x[:, -1, :]

        # 最终分类
        return self.kan(x)
