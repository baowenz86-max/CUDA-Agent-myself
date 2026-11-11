import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.sync_batchnorm = nn.SyncBatchNorm(num_features)
        self.num_classes = num_classes

    def forward(self, x, elements):
        x = cuda_extension.sync_batchnorm_forward(
            x,
            self.sync_batchnorm.weight,
            self.sync_batchnorm.bias,
            self.sync_batchnorm.running_mean,
            self.sync_batchnorm.running_var,
            self.sync_batchnorm.eps,
        )
        mask = cuda_extension.isin_forward(x, elements)
        return x, mask

    def debug_forward(self, x, elements):
        x = cuda_extension.sync_batchnorm_forward(
            x,
            self.sync_batchnorm.weight,
            self.sync_batchnorm.bias,
            self.sync_batchnorm.running_mean,
            self.sync_batchnorm.running_var,
            self.sync_batchnorm.eps,
        )
        yield "step_001", x
        mask = cuda_extension.isin_forward(x, elements)
        yield "step_002", mask
        yield "step_003", (x, mask)
