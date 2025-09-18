import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (32, 18)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, 32, 18)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, input_dim, seq_len=32, num_classes=10, nhead=2, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=input_dim, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (B, 32, 18)
        x = self.pos_encoder(x)  # 加上位置编码
        x = self.transformer(x)  # (B, 32, 18)

        # 取平均池化表示
        x = x.mean(dim=1)  # (B, 18)

        out = self.classifier(x)  # (B, 10)
        return out
