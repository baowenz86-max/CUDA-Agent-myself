import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, eps):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_channels, num_channels)
        self.eps = eps

    def forward(self, x):
        # Move parameters to CUDA if necessary; assume input is CUDA float32 contiguous
        weight_cuda = self.group_norm.weight.detach().to(x.device).contiguous()
        bias_cuda = self.group_norm.bias.detach().to(x.device).contiguous()
        x = cuda_extension.group_norm_forward(
            x.contiguous(),
            weight_cuda,
            bias_cuda,
            self.group_norm.num_groups,
            self.eps
        )
        x = cuda_extension.log_softmax_dim1_forward(x)
        return x

    def debug_forward(self, x):
        weight_cuda = self.group_norm.weight.detach().to(x.device).contiguous()
        bias_cuda = self.group_norm.bias.detach().to(x.device).contiguous()
        gn_out = cuda_extension.group_norm_forward(
            x.contiguous(),
            weight_cuda,
            bias_cuda,
            self.group_norm.num_groups,
            self.eps
        )
        yield 'step_001', gn_out
        ls_out = cuda_extension.log_softmax_dim1_forward(gn_out)
        yield 'step_002', ls_out
        yield 'step_003', ls_out
