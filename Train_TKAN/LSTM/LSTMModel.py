import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, rnn_unit, output_size, dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, rnn_unit, batch_first=True, dropout=0)
        self.lstm2 = nn.LSTM(rnn_unit, output_size, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        return x