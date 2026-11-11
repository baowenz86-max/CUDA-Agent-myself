import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        # 创建与参考 nn.RNN 相同的参数，保证 state_dict 键一致
        self.rnn = nn.Module()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            setattr(self.rnn, f'weight_ih_l{layer}', nn.Parameter(torch.empty(hidden_size, layer_input_size)))
            setattr(self.rnn, f'weight_hh_l{layer}', nn.Parameter(torch.empty(hidden_size, hidden_size)))
            if bias:
                setattr(self.rnn, f'bias_ih_l{layer}', nn.Parameter(torch.empty(hidden_size)))
                setattr(self.rnn, f'bias_hh_l{layer}', nn.Parameter(torch.empty(hidden_size)))
            else:
                setattr(self.rnn, f'bias_ih_l{layer}', None)
                setattr(self.rnn, f'bias_hh_l{layer}', None)

    def forward(self, x: torch.Tensor):
        return cuda_extension.rnn_forward(
            x,
            self.rnn.weight_ih_l0,
            self.rnn.weight_hh_l0,
            self.rnn.bias_ih_l0,
            self.rnn.bias_hh_l0,
            self.rnn.weight_ih_l1,
            self.rnn.weight_hh_l1,
            self.rnn.bias_ih_l1,
            self.rnn.bias_hh_l1,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias
        )

    def debug_forward(self, x: torch.Tensor):
        out = self.forward(x)
        yield "step_001", out
