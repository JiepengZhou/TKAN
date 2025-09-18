import torch.nn as nn
import torchvision.models as models

class PureDenseNet(nn.Module):
    def __init__(self, num_classes=10, flrn=5):
        super(PureDenseNet, self).__init__()

        # 加载 DenseNet
        self.densenet = models.densenet121(weights=models.densenet.DenseNet121_Weights.IMAGENET1K_V1)

        # 获取 DenseNet 输出维度
        self.feature_dim = self.densenet.classifier.in_features

        # 修改 DenseNet 的分类头为全连接层
        self.densenet.classifier = nn.Linear(self.feature_dim, num_classes)

        # 冻结前 flrn 层参数
        children = list(self.densenet.features.children())
        for child in children[:flrn]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 直接使用 DenseNet 进行特征提取和分类
        x = self.densenet(x)
        return x