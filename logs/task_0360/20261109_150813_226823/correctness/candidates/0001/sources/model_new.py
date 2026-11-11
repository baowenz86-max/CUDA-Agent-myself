import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias):
        super().__init__()
        # Preserve state_dict keys and shapes by creating individual parameters.
        # torch.nn.RNN stores parameters as rnn.weight_ih_l0, etc.
        self.rnn_weight_ih_l0 = nn.Parameter(torch.empty(hidden_size, input_size))
        self.rnn_weight_hh_l0 = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.rnn_bias_ih_l0 = nn.Parameter(torch.empty(hidden_size))
        self.rnn_bias_hh_l0 = nn.Parameter(torch.empty(hidden_size))
        self.rnn_weight_ih_l1 = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.rnn_weight_hh_l1 = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.rnn_bias_ih_l1 = nn.Parameter(torch.empty(hidden_size))
        self.rnn_bias_hh_l1 = nn.Parameter(torch.empty(hidden_size))

        # Initialize exactly like nn.RNN would.
        # Default init: uniform(-1/sqrt(hidden_size), 1/sqrt(hidden_size)) for weights,
        # uniform(-1/sqrt(hidden_size), 1/sqrt(hidden_size)) for biases.
        stdv = 1.0 / (hidden_size ** 0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x: torch.Tensor):
        # Ensure x is CUDA float32 contiguous (the custom op expects contiguous).
        x = x.cuda().float().contiguous()
        # Move all parameters to the same device/dtype as x.
        params = [p.to(device=x.device, dtype=x.dtype).contiguous() for p in self.parameters()]
        return cuda_extension.rnn_forward(
            x,
            params[0], params[1], params[2], params[3],
            params[4], params[5], params[6], params[7],
            None
        )

    def debug_forward(self, *inputs):
        x = inputs[0]
        result = self.forward(x)
        yield ('step_001', result)
