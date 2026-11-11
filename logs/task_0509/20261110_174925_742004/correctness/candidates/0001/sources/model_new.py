import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        # Preserve state_dict keys exactly: linear1.weight, linear1.bias, linear2.weight, linear2.bias,
        # norm.weight, norm.bias, norm.running_mean, norm.running_var, norm.num_batches_tracked
        self.linear1_weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.linear1_bias = nn.Parameter(torch.empty(hidden_size))
        self.linear2_weight = nn.Parameter(torch.empty(input_size, hidden_size))
        self.linear2_bias = nn.Parameter(torch.empty(input_size))
        self.norm_weight = nn.Parameter(torch.empty(input_size))
        self.norm_bias = nn.Parameter(torch.empty(input_size))
        self.norm_running_mean = nn.Parameter(torch.empty(input_size), requires_grad=False)
        self.norm_running_var = nn.Parameter(torch.empty(input_size), requires_grad=False)
        self.norm_num_batches_tracked = nn.Parameter(torch.zeros((), dtype=torch.int64), requires_grad=False)

        # Initialize parameters to match reference defaults (nn.Linear and BatchNorm1d)
        # Linear: kaiming_uniform with a=sqrt(5), bias uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
        # BatchNorm1d: weight=1, bias=0, running_mean=0, running_var=1, num_batches_tracked=0
        nn.init.kaiming_uniform_(self.linear1_weight, a=5 ** 0.5)
        bound = 1 / (input_size ** 0.5)
        nn.init.uniform_(self.linear1_bias, -bound, bound)
        nn.init.kaiming_uniform_(self.linear2_weight, a=5 ** 0.5)
        bound = 1 / (hidden_size ** 0.5)
        nn.init.uniform_(self.linear2_bias, -bound, bound)
        nn.init.ones_(self.norm_weight)
        nn.init.zeros_(self.norm_bias)
        nn.init.zeros_(self.norm_running_mean)
        nn.init.ones_(self.norm_running_var)

        # Store eps and momentum for batch_norm1d_forward call
        self.eps = 1e-05
        self.momentum = 0.1

    def forward(self, x):
        # Step 1: linear1
        x = cuda_extension.linear_forward(x, self.linear1_weight, self.linear1_bias)
        # Step 2: sigmoid
        x = cuda_extension.sigmoid_forward(x)
        # Step 3: linear2
        x = cuda_extension.linear_forward(x, self.linear2_weight, self.linear2_bias)
        # Step 4: batch norm (inference)
        x = cuda_extension.batch_norm1d_forward(
            x,
            self.norm_weight,
            self.norm_bias,
            self.norm_running_mean,
            self.norm_running_var,
            self.eps,
            self.momentum,
        )
        return x

    def debug_forward(self, x):
        # Step 1: linear1
        x = cuda_extension.linear_forward(x, self.linear1_weight, self.linear1_bias)
        yield "step_001", x
        # Step 2: sigmoid
        x = cuda_extension.sigmoid_forward(x)
        yield "step_002", x
        # Step 3: linear2
        x = cuda_extension.linear_forward(x, self.linear2_weight, self.linear2_bias)
        yield "step_003", x
        # Step 4: batch norm
        x = cuda_extension.batch_norm1d_forward(
            x,
            self.norm_weight,
            self.norm_bias,
            self.norm_running_mean,
            self.norm_running_var,
            self.eps,
            self.momentum,
        )
        yield "step_004", x
        # Final return
        yield "step_005", x
