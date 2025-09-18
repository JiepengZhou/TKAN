import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, feature_num, d_model=128, num_heads=4, num_layers=2, dim_feedforward=256):
        super(Transformer, self).__init__()
        self.input_proj = nn.Linear(feature_num, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))  # 假设最大序列长度为 1000
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, feature_num)

    def forward(self, x):
        batch_size, time_step, feature_num = x.shape
        x = self.input_proj(x) + self.pos_encoder[:, :time_step, :]
        x = self.transformer_encoder(x)
        x = self.output_proj(x)
        return x