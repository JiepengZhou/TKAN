import torch.nn as nn
import torch
from MyKan import KAN

class TKAN(nn.Module):
    def __init__(
        self,
        input_size: int, # 输入维度
        hidden_size: int, # 隐层神经元数量
        kan_layers_hidden: list, # Kan模型的输入
        output_size: int, # 输出维度
        bidirectional: bool = False,
    ):
        super(TKAN, self).__init__()
        # LSTM 层，两层LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=2, # 这里设置了两层LSTM
            batch_first=True,
            dropout=0
        )
        # KAN 层
        self.kan = KAN(layers_hidden=kan_layers_hidden)
        # 输出层
        self.output_layer = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x: torch.Tensor):
        # LSTM 前向传播
        lstm_output, _ = self.lstm(x)  # lstm_output 形状: [batch_size, timestep, hidden_size]
        # print(f"lstm_output shape: {lstm_output.shape}")
        # KAN 前向传播， 实际上kan就是linear，只不过换成了线性函数的组合，所以他的输出可以直接作为模型的输出
        # 将LSTM的输出与原x组合，让Kan更优地学习，希望效果能变好
        kan_input = torch.cat([x, lstm_output], dim=-1)  # 变成 [batch_size, time_step, feature_num + hidden_dim]
        # kan
        kan_output = self.kan(kan_input)  # kan_output 形状: [batch_size, timestep, feature_num]
        # print(f"kan_out shape: {kan_output.shape}")
        return kan_output