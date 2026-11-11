import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers):
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Keep original submodules to preserve state_dict keys and initialization logic.
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        # x is expected time-major: [seq_len, batch_size, input_size]
        batch_size = x.size(1)
        h0 = cuda_extension.zeros_like_alloc(self.num_layers, batch_size, self.hidden_size, x.device.index)
        c0 = cuda_extension.zeros_like_alloc(self.num_layers, batch_size, self.hidden_size, x.device.index)

        lstm_out = cuda_extension.lstm_forward(
            x,
            h0,
            c0,
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l1,
            self.lstm.bias_ih_l1,
            self.lstm.bias_hh_l1,
            self.num_layers,
            self.hidden_size
        )
        out = lstm_out[0]

        out = cuda_extension.linear_fc(out[:, -1, :], self.fc.weight, self.fc.bias)
        return out

    def debug_forward(self, *inputs):
        x = inputs[0]
        batch_size = x.size(1)
        # step_001
        h0 = cuda_extension.zeros_like_alloc(self.num_layers, batch_size, self.hidden_size, x.device.index)
        yield 'step_001', h0
        # step_002
        c0 = cuda_extension.zeros_like_alloc(self.num_layers, batch_size, self.hidden_size, x.device.index)
        yield 'step_002', c0
        # step_003
        lstm_out = cuda_extension.lstm_forward(
            x,
            h0,
            c0,
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l1,
            self.lstm.bias_ih_l1,
            self.lstm.bias_hh_l1,
            self.num_layers,
            self.hidden_size
        )
        out = lstm_out[0]
        yield 'step_003', lstm_out
        # step_004
        out = cuda_extension.linear_fc(out[:, -1, :], self.fc.weight, self.fc.bias)
        yield 'step_004', out
        # step_005
        yield 'step_005', out
