import torch
import torch.nn as nn

class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义 SRU 层
        self.sru_cells = nn.ModuleList([
            nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        
        for t in range(seq_length):
            xt = x[:, t, :]  # 当前时间步的数据
            for i, sru_cell in enumerate(self.sru_cells):
                h[i] = sru_cell(xt, h[i])
                xt = h[i]  # 当前层的输出作为下一层的输入

        x = self.fc(h[-1])  # 使用最后一层的状态
        return x