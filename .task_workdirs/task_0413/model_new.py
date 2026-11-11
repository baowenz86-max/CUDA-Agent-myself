import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.sync_bn = nn.SyncBatchNorm(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x_max, _ = cuda_extension.max_dim_reduce(x, dim=1, keepdim=True)
        x = cuda_extension.sync_bn_inference(
            x,
            self.sync_bn.weight,
            self.sync_bn.bias,
            self.sync_bn.running_mean,
            self.sync_bn.running_var,
            self.sync_bn.eps,
        )
        return x, x_max

    def debug_forward(self, x):
        x_max, indices = cuda_extension.max_dim_reduce(x, dim=1, keepdim=True)
        yield "step_001", (x_max, indices)
        x = cuda_extension.sync_bn_inference(
            x,
            self.sync_bn.weight,
            self.sync_bn.bias,
            self.sync_bn.running_mean,
            self.sync_bn.running_var,
            self.sync_bn.eps,
        )
        yield "step_002", x
        yield "step_003", (x, x_max)
