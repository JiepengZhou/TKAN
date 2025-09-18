import torch
import torch.nn as nn
from MyKan import KAN

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, input_dim, seq_len=32, num_classes=10, nhead=2, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=input_dim, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True)

        # 假设你已定义 KAN 类
        self.kan_lstm = KAN([128, 64, num_classes])
        self.kan_transformer = KAN([input_dim, 64, num_classes])

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B, 32, input_dim)
        x = self.pos_encoder(x)
        x_transformer = self.transformer(x)  # (B, 32, input_dim)

        # 平均池化得到 transformer 表征
        x_mean = x_transformer.mean(dim=1)  # (B, input_dim)
        transformer_res = self.kan_transformer(x_mean)

        # LSTM -> 最后时间步表示 -> KAN
        lstm_out, _ = self.lstm(x_transformer)  # (B, 32, 128)
        last_step = lstm_out[:, -1, :]  # (B, 128)
        lstm_res = self.kan_lstm(last_step)  # (B, 10)

        # 融合两个专家输出
        out = transformer_res + self.alpha * (lstm_res - transformer_res)
        return out
