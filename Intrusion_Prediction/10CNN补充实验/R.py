import torch.nn as nn
import torchvision.models as models

class PureResNet(nn.Module):
    def __init__(self, num_classes=10, flrn=5):
        super(PureResNet, self).__init__()
        
        # 使用 ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 获取 ResNet 的输出特征维度
        self.feature_dim = self.resnet.fc.in_features
        
        # 修改 ResNet 的分类头为全连接层
        self.resnet.fc = nn.Linear(self.feature_dim, num_classes)

        # 冻结部分层（前 flrn 层）
        children = list(self.resnet.children())[:-2]  # 除去分类头和全连接层部分
        for child in children[:flrn]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        # ResNet 提取特征并直接用于分类
        x = self.resnet(x)
        return x