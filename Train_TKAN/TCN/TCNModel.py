import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2  # 确保时间步不变

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        
        # 残差连接，如果通道数不同需要 1x1 卷积匹配
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.relu(x + res)

class TCN(nn.Module):
    def __init__(self, feature_num, num_channels=[32, 64, 32], kernel_size=3):
        super(TCN, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            in_channels = feature_num if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation = 2 ** i  # 指数增长，适合短时间序列
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation))
        self.tcn = nn.Sequential(*layers)
        self.output_proj = nn.Linear(num_channels[-1], feature_num)  # 线性层输出
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 变为 [batch_size, feature_num, time_step]
        x = self.tcn(x)
        x = x.permute(0, 2, 1)  # 变回 [batch_size, time_step, feature_num]
        return self.output_proj(x)