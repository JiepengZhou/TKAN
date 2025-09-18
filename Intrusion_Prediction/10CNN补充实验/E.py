import torch.nn as nn
import torchvision.models as models

class PureEfficientnet(nn.Module):
    def __init__(self, num_classes=10, flrn=5):
        super(PureEfficientnet, self).__init__()
        
        # 使用 EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # 获取 EfficientNet 的输出特征维度
        self.feature_dim = self.efficientnet.classifier[1].in_features
        
        # 修改 EfficientNet 的分类头为全连接层
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, num_classes)
        )

        # 冻结部分层（前 flrn 层）
        children = list(self.efficientnet.features.children())
        for child in children[:flrn]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        # EfficientNet 提取特征并直接用于分类
        x = self.efficientnet(x)
        return x