import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, feature_num, hidden_dim=128, num_layers=2):
        super(MLP, self).__init__()
        layers = []
        input_dim = feature_num
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, feature_num))  # 输出与输入维度相同
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, time_step, feature_num = x.shape
        x = x.view(-1, feature_num)  # 变形为 [batch_size * time_step, feature_num]
        x = self.mlp(x)
        x = x.view(batch_size, time_step, feature_num)  # 还原形状
        return x