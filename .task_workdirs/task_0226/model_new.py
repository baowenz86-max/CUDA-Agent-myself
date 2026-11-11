import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features):
        super(ModelNew, self).__init__()
        self.batchnorm1d = torch.nn.BatchNorm1d(num_features)

    def forward(self, x):
        return cuda_extension.batchnorm1d_inference_forward(
            x,
            self.batchnorm1d.weight,
            self.batchnorm1d.bias,
            self.batchnorm1d.running_mean,
            self.batchnorm1d.running_var,
            self.batchnorm1d.eps,
        )

    def debug_forward(self, *inputs):
        if len(inputs) != 1:
            raise ValueError('debug_forward expects exactly one input tensor')
        x = inputs[0]
        out = cuda_extension.batchnorm1d_inference_forward(
            x,
            self.batchnorm1d.weight,
            self.batchnorm1d.bias,
            self.batchnorm1d.running_mean,
            self.batchnorm1d.running_var,
            self.batchnorm1d.eps,
        )
        yield 'step_001', out
