import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a submodule named 'lstm' to match state_dict keys
        self.lstm = nn.Module()
        for layer in range(num_layers):
            input_dim = input_size if layer == 0 else hidden_size
            setattr(self.lstm, f'weight_ih_l{layer}', nn.Parameter(torch.empty(4 * hidden_size, input_dim)))
            setattr(self.lstm, f'weight_hh_l{layer}', nn.Parameter(torch.empty(4 * hidden_size, hidden_size)))
            setattr(self.lstm, f'bias_ih_l{layer}', nn.Parameter(torch.empty(4 * hidden_size)))
            setattr(self.lstm, f'bias_hh_l{layer}', nn.Parameter(torch.empty(4 * hidden_size)))

        # Initialize parameters (same as PyTorch LSTM default initialization)
        stdv = 1.0 / (hidden_size ** 0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x):
        device = x.device
        weights = [getattr(self.lstm, f'weight_ih_l{l}').to(device) for l in range(self.num_layers)]
        weights += [getattr(self.lstm, f'weight_hh_l{l}').to(device) for l in range(self.num_layers)]
        biases = [getattr(self.lstm, f'bias_ih_l{l}').to(device) for l in range(self.num_layers)]
        biases += [getattr(self.lstm, f'bias_hh_l{l}').to(device) for l in range(self.num_layers)]

        outputs, hidden, cell = cuda_extension.lstm_forward(
            x, *weights, *biases,
            self.input_size, self.hidden_size, self.num_layers
        )

        result = cuda_extension.cat_hidden_cell(hidden, cell)
        return result

    def debug_forward(self, x):
        device = x.device
        weights = [getattr(self.lstm, f'weight_ih_l{l}').to(device) for l in range(self.num_layers)]
        weights += [getattr(self.lstm, f'weight_hh_l{l}').to(device) for l in range(self.num_layers)]
        biases = [getattr(self.lstm, f'bias_ih_l{l}').to(device) for l in range(self.num_layers)]
        biases += [getattr(self.lstm, f'bias_hh_l{l}').to(device) for l in range(self.num_layers)]

        outputs, hidden, cell = cuda_extension.lstm_forward(
            x, *weights, *biases,
            self.input_size, self.hidden_size, self.num_layers
        )
        yield 'step_001', (outputs, (hidden, cell))

        result = cuda_extension.cat_hidden_cell(hidden, cell)
        yield 'step_002', result
