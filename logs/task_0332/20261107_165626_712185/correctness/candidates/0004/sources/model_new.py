import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, pooling_length):
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(pooling_length)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * pooling_length, output_size)
        self.pooling_length = pooling_length

    def forward(self, x):
        x = cuda_extension.adaptive_avg_pool1d(x, self.pooling_length)
        x = cuda_extension.transpose_1_2(x)
        x, _hidden = cuda_extension.lstm_forward(
            x,
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l1,
            self.lstm.bias_ih_l1,
            self.lstm.bias_hh_l1,
            self.hidden_size,
            self.num_layers,
        )
        x = cuda_extension.reshape_2d(x, x.size(0), -1)
        x = cuda_extension.linear(x, self.fc.weight, self.fc.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.adaptive_avg_pool1d(x, self.pooling_length)
        yield "step_001", x
        x = cuda_extension.transpose_1_2(x)
        yield "step_002", x
        x, _hidden = cuda_extension.lstm_forward(
            x,
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l1,
            self.lstm.bias_ih_l1,
            self.lstm.bias_hh_l1,
            self.hidden_size,
            self.num_layers,
        )
        yield "step_003", (x, _hidden)
        x = cuda_extension.reshape_2d(x, x.size(0), -1)
        yield "step_004", x
        x = cuda_extension.linear(x, self.fc.weight, self.fc.bias)
        yield "step_005", x
        yield "step_006", x
