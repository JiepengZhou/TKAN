import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 调整因子
        self.reduction = reduction  # 损失的归约方式

    def forward(self, inputs, targets):
        """
        inputs: [batch_size, num_classes] - 模型的预测输出（logits）
        targets: [batch_size, num_classes] - 真实标签（one-hot 形式）
        """
        if targets.dim() == 1:
            targets = torch.eye(inputs.size(1), device=inputs.device)[targets]  # 转换为 one-hot
            
        probs = torch.softmax(inputs, dim=1)  # 获取类别概率
        pt = (probs * targets).sum(dim=1)  # 计算 Mixup/CutMix 兼容的 pt
        ce_loss = -(targets * torch.log(probs + 1e-8)).sum(dim=1)  # 交叉熵损失

        if self.alpha is not None:
            alpha_t = (self.alpha * targets).sum(dim=1)  # 计算 alpha
            loss = alpha_t * (1 - pt) ** self.gamma * ce_loss  # Focal Loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

