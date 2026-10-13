import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias_weight, bias_lstm, clamp_min, clamp_max):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias_weight)
        self.bias = nn.Parameter(torch.tensor(bias_lstm))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = cuda_extension.layer_norm_default(x, x.shape[-1], 1e-05)
        x = cuda_extension.layer_norm_affine(x, self.layer_norm.weight, self.layer_norm.bias, 1e-05)
        x = cuda_extension.nan_to_num(x)
        x, (hn, cn) = cuda_extension.lstm(
            x,
            self.lstm.weight_ih_l0, self.lstm.weight_hh_l0, self.lstm.bias_ih_l0, self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1, self.lstm.weight_hh_l1, self.lstm.bias_ih_l1, self.lstm.bias_hh_l1,
            self.lstm.num_layers
        )
        x = cuda_extension.add_bias(x, self.bias)
        x = cuda_extension.clamp(x, self.clamp_min, self.clamp_max)
        return x

    def debug_forward(self, x):
        x = cuda_extension.layer_norm_default(x, x.shape[-1], 1e-05)
        yield "step_001", x
        x = cuda_extension.layer_norm_affine(x, self.layer_norm.weight, self.layer_norm.bias, 1e-05)
        yield "step_002", x
        x = cuda_extension.nan_to_num(x)
        yield "step_003", x
        x, (hn, cn) = cuda_extension.lstm(
            x,
            self.lstm.weight_ih_l0, self.lstm.weight_hh_l0, self.lstm.bias_ih_l0, self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1, self.lstm.weight_hh_l1, self.lstm.bias_ih_l1, self.lstm.bias_hh_l1,
            self.lstm.num_layers
        )
        yield "step_004", (x, (hn, cn))
        x = cuda_extension.add_bias(x, self.bias)
        yield "step_005", x
        x = cuda_extension.clamp(x, self.clamp_min, self.clamp_max)
        yield "step_006", x
        yield "step_007", x
