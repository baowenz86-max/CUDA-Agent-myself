import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_classes, bucket_boundaries_shape):
        super().__init__()
        self.num_classes = num_classes
        self.bucket_boundaries = nn.Parameter(torch.randn(bucket_boundaries_shape))

    def forward(self, x, values):
        x = cuda_extension.one_hot_forward(x, self.num_classes)
        x = cuda_extension.maximum_scalar_forward(x, 0.0)
        x = cuda_extension.bucketize_forward(values, self.bucket_boundaries)
        return x

    def debug_forward(self, x, values):
        one_hot = cuda_extension.one_hot_forward(x, self.num_classes)
        yield 'step_001', one_hot

        maximum = cuda_extension.maximum_scalar_forward(one_hot, 0.0)
        yield 'step_002', maximum

        bucketize = cuda_extension.bucketize_forward(values, self.bucket_boundaries)
        yield 'step_003', bucketize

        yield 'step_004', bucketize
