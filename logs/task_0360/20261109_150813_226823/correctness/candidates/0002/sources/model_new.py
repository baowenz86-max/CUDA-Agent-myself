import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, bias=bias)

    def forward(self, x: torch.Tensor):
        x = x.cuda().float().contiguous()
        params = []
        # 按照 RNN 参数顺序：weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0,
        # weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1
        for name in ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0',
                     'weight_ih_l1', 'weight_hh_l1', 'bias_ih_l1', 'bias_hh_l1']:
            p = getattr(self.rnn, name)
            params.append(p.to(device=x.device, dtype=x.dtype).contiguous())
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
