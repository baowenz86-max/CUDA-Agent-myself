import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = cuda_extension.lstm_forward(
            x,
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l1,
            self.lstm.bias_ih_l1,
            self.lstm.bias_hh_l1,
            self.lstm.num_layers,
            self.lstm.batch_first,
            self.training
        )
        result = cuda_extension.cat_along_dim(hidden, cell, 1)
        return result

    def debug_forward(self, x):
        outputs, (hidden, cell) = cuda_extension.lstm_forward(
            x,
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l1,
            self.lstm.bias_ih_l1,
            self.lstm.bias_hh_l1,
            self.lstm.num_layers,
            self.lstm.batch_first,
            self.training
        )
        yield 'step_001', (outputs, (hidden, cell))
        result = cuda_extension.cat_along_dim(hidden, cell, 1)
        yield 'step_002', result
