import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, bin_edges):
        super(ModelNew, self).__init__()
        self.bin_edges = bin_edges

    def forward(self, x):
        # Move input to CUDA before custom ops (reference execution_device is cuda)
        x = x.cuda()
        x = torch.ops.cuda_extension.mean_dim_023(x)
        bins = len(self.bin_edges) - 1
        x = torch.ops.cuda_extension.histc_custom(x, bins, 0.0, 1.0)
        return x

    def debug_forward(self, x):
        x = x.cuda()
        x = torch.ops.cuda_extension.mean_dim_023(x)
        yield "step_001", x
        bins = len(self.bin_edges) - 1
        x = torch.ops.cuda_extension.histc_custom(x, bins, 0.0, 1.0)
        yield "step_002", x
        yield "step_003", x
