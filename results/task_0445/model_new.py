import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps, affine):
        super().__init__()
        self.sync_batch_norm = nn.SyncBatchNorm(num_features, eps, affine)

    def forward(self, x):
        x = cuda_extension.absolute(x)
        x = cuda_extension.sync_batch_norm_eval(
            x,
            self.sync_batch_norm.weight,
            self.sync_batch_norm.bias,
            self.sync_batch_norm.running_mean,
            self.sync_batch_norm.running_var,
            self.sync_batch_norm.eps,
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.absolute(x)
        yield 'step_001', x
        x = cuda_extension.sync_batch_norm_eval(
            x,
            self.sync_batch_norm.weight,
            self.sync_batch_norm.bias,
            self.sync_batch_norm.running_mean,
            self.sync_batch_norm.running_var,
            self.sync_batch_norm.eps,
        )
        yield 'step_002', x
        yield 'step_003', x
