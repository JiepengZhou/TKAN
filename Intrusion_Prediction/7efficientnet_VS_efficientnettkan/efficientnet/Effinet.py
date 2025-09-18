import torch
import torch.nn as nn
import torchvision.models as models

def create_efficientnet(num_classes=10, freeze_layers=5):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)  # 加载预训练模型
    
    # 调整第一层的输入通道数，使其适应 3x32x32 输入
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 冻结前 5 层参数
    frozen_count = 0
    for layer in model.features[:freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False
            frozen_count += 1
    print(f"Frozen {frozen_count} layers")
    
    # 修改分类头以适配 num_classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model
