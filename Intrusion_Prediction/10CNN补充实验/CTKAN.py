import torch
import torch.nn as nn
import torchvision.models as models
from MyKan import KAN 

class CTKAN(nn.Module):
    def __init__(self, num_classes=10, flrn=5):
        super(CTKAN, self).__init__()

        # 使用 ConvNeXt
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # 修改 ConvNeXt 的第一个卷积层以支持单通道输入
        original_conv1 = self.convnext.features[0][0]  # 获取第一个卷积层
        self.convnext.features[0][0] = nn.Conv2d(
            in_channels=1,  # 单通道输入
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )

        # 获取 ConvNeXt 的输出特征维度
        self.feature_dim = self.convnext.classifier[2].in_features

        # 去掉 ConvNeXt 的分类头（只保留特征提取部分）
        self.convnext.classifier = nn.Identity()

        # 冻结前 flrn 层参数
        children = list(self.convnext.features.children())
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
        x = self.convnext(x)  # (B, feature_dim)

        # 送入 LSTM，注意 reshape 成 (B, 1, feature_dim)
        x = x.view(batch_size, 1, self.feature_dim)
        x, _ = self.lstm(x)

        # 取 LSTM 的最后一步
        x = x[:, -1, :]

        # 最终分类
        return self.kan(x)