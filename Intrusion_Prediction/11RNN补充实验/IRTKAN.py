import torch
import torch.nn as nn
from MyKan import KAN 

class IRTKAN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, seq_length):
        super(IRTKAN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # 定义 IndRNN 层
        self.rnn_layers = nn.ModuleList([
            nn.RNN(input_size if i == 0 else hidden_size, hidden_size, batch_first=True, nonlinearity='relu')
            for i in range(num_layers)
        ])
        
        # LSTM 处理全局特征
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        # KAN 结构（此处简化为全连接层）
        self.kan = KAN([256, 64, num_classes])

    def forward(self, x):
        h = [torch.zeros(1, x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, h[i] = rnn_layer(x, h[i])
            
        # 获取 IndRNN 的最后一个时间步的输出
        x_last = x[:, -1, :].unsqueeze(1)  # 形状为 (batch_size, 1, hidden_size)
        
        # 输入 LSTM 层
        lstm_out, _ = self.lstm(x_last)  # LSTM 的输出形状为 (batch_size, 1, hidden_size)
        
        # 提取 LSTM 的最后一个时间步的输出
        lstm_last = lstm_out[:, -1, :]  # 形状为 (batch_size, hidden_size)

        x = self.kan(lstm_last)
        
        return x