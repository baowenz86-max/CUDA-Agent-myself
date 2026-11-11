import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        x = cuda_extension.linear_forward(x, self.linear1.weight, self.linear1.bias)
        x = cuda_extension.sigmoid_forward(x)
        x = cuda_extension.linear_forward(x, self.linear2.weight, self.linear2.bias)
        x = cuda_extension.batch_norm1d_forward(
            x,
            self.norm.weight,
            self.norm.bias,
            self.norm.running_mean,
            self.norm.running_var,
            self.norm.eps,
            self.norm.momentum,
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.linear_forward(x, self.linear1.weight, self.linear1.bias)
        yield "step_001", x
        x = cuda_extension.sigmoid_forward(x)
        yield "step_002", x
        x = cuda_extension.linear_forward(x, self.linear2.weight, self.linear2.bias)
        yield "step_003", x
        x = cuda_extension.batch_norm1d_forward(
            x,
            self.norm.weight,
            self.norm.bias,
            self.norm.running_mean,
            self.norm.running_var,
            self.norm.eps,
            self.norm.momentum,
        )
        yield "step_004", x
        yield "step_005", x
