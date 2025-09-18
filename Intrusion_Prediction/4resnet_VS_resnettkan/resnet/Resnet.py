import torch
import torch.nn as nn
import torchvision.models as models

def create_resnet18(num_classes=10, freeze_layers=5):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 加载预训练模型
    
    # 调整第一层的输入通道数，使其适应 3x32x32 输入
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # CIFAR-10 尺寸较小，去掉 maxpool 层
    
    # 冻结前 5 层参数
    layers = [model.conv1, model.bn1, model.relu, model.layer1]
    frozen_count = 0
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False
            frozen_count += 1
    print(f"Frozen {frozen_count} layers")
    
    # 修改最后的全连接层以适配 num_classes
    model.fc = nn.Linear(512, num_classes)
    
    return model
