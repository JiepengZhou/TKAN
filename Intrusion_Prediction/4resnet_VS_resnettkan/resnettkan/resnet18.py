import torch.nn as nn
from torchvision import models 

class Resnet(nn.Module):
    def __init__(self, num_classes=10, freeze_until_layer=5):
        # 这里默认从第五层修改权重，上面五层冻结
        super(Resnet, self).__init__()

        # 载入 ResNet18 预训练模型
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层 (fc)

        # 获取 ResNet 的输出特征维度
        self.feature_dim = 512  # ResNet18 最后一个全局平均池化层的输出通道数

        # 解冻部分层
        # 默认冻结所有层
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 解冻从指定的层开始，freeze_until_layer为解冻的起始层索引
        layers = list(self.resnet.children())
        for i in range(freeze_until_layer, len(layers)):
            for param in layers[i].parameters():
                param.requires_grad = True

        # 添加全连接层用于分类
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.resnet(x)  # ResNet 提取特征
        x = x.view(x.size(0), -1)  # 展平成 (batch_size, feature_dim)
        x = self.fc(x)  # 通过全连接层分类
        return x