import torch.nn as nn
import torchvision.models as models
import torch

class PureConvNext(nn.Module):
    def __init__(self, num_classes=10, flrn=5):
        super(PureConvNext, self).__init__()

        # 使用 ConvNeXt
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # 获取 ConvNeXt 的输出特征维度
        self.feature_dim = self.convnext.classifier[2].in_features

        # 修改 ConvNeXt 的分类头为全连接层，并保留 Flatten 操作
        self.convnext.classifier = nn.Sequential(
            nn.Flatten(),  # 展平操作
            nn.Linear(self.feature_dim, num_classes)  # 全连接层
        )

        # 冻结前 flrn 层参数
        children = list(self.convnext.features.children())
        for child in children[:flrn]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 直接使用 ConvNeXt 进行特征提取和分类
        x = self.convnext(x)
        return x