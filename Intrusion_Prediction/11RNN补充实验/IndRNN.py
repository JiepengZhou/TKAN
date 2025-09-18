import torch
import torch.nn as nn

class IndRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, seq_length):
        super(IndRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # 定义 IndRNN 层
        self.rnn_layers = nn.ModuleList([
            nn.RNN(input_size if i == 0 else hidden_size, hidden_size, batch_first=True, nonlinearity='relu')
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h = [torch.zeros(1, x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, h[i] = rnn_layer(x, h[i])
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)
        return x