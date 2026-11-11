import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # Preserve exact state_dict structure of reference BatchNorm1d
        self.batch_norm = nn.Module()
        self.batch_norm.weight = nn.Parameter(torch.ones(num_features))
        self.batch_norm.bias = nn.Parameter(torch.zeros(num_features))
        self.batch_norm.register_buffer('running_mean', torch.zeros(num_features))
        self.batch_norm.register_buffer('running_var', torch.ones(num_features))
        self.batch_norm.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Ensure input is on CUDA
        if not x.is_cuda:
            x = x.cuda()
        # Ensure parameters are on CUDA (in case model is moved after init)
        if not self.batch_norm.weight.is_cuda:
            self.batch_norm.weight = self.batch_norm.weight.cuda()
            self.batch_norm.bias = self.batch_norm.bias.cuda()
            self.batch_norm.running_mean = self.batch_norm.running_mean.cuda()
            self.batch_norm.running_var = self.batch_norm.running_var.cuda()
            # num_batches_tracked is a scalar long, move as well
            self.batch_norm.num_batches_tracked = self.batch_norm.num_batches_tracked.cuda()

        # Step 1: ReLU
        x = cuda_extension.relu_forward(x)
        # Step 2: BatchNorm1d in eval mode using running stats
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.eps,
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        if not self.batch_norm.weight.is_cuda:
            self.batch_norm.weight = self.batch_norm.weight.cuda()
            self.batch_norm.bias = self.batch_norm.bias.cuda()
            self.batch_norm.running_mean = self.batch_norm.running_mean.cuda()
            self.batch_norm.running_var = self.batch_norm.running_var.cuda()
            self.batch_norm.num_batches_tracked = self.batch_norm.num_batches_tracked.cuda()
        # checkpoint step_001
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        # checkpoint step_002
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.eps,
        )
        yield 'step_002', x
        # checkpoint step_003
        yield 'step_003', x
