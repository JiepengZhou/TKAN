import torch
import torch.nn as nn
from MyKan import KAN

class SRTKAN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SRTKAN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义 SRU 层
        self.sru_cells = nn.ModuleList([
            nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # LSTM 处理全局特征
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        # KAN 结构（此处简化为全连接层）
        self.kan = KAN([256, 64, num_classes])

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        
        for t in range(seq_length):
            xt = x[:, t, :]  # 当前时间步的数据
            for i, sru_cell in enumerate(self.sru_cells):
                h[i] = sru_cell(xt, h[i])
                xt = h[i]  # 当前层的输出作为下一层的输入

        # 获取 SRU 最后一层的隐藏状态
        sru_last_hidden = h[-1].unsqueeze(1)  # 形状为 (batch_size, 1, hidden_size)
        
        # 输入到 LSTM 层
        lstm_out, _ = self.lstm(sru_last_hidden)  # LSTM 的输出形状为 (batch_size, 1, hidden_size)
        
        # 提取 LSTM 的最后一个时间步的输出
        lstm_last = lstm_out[:, -1, :]  # 形状为 (batch_size, hidden_size)

        x = self.kan(lstm_last)
        return x