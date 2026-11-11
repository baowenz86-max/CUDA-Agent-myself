import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, mean_value):
        super().__init__()
        self.mean = mean_value

    def forward(self, x):
        x = cuda_extension.custom_reduce_mean_channel(x)
        x = cuda_extension.custom_add_scalar(x, self.mean)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_reduce_mean_channel(x)
        yield 'step_001', x
        x = cuda_extension.custom_add_scalar(x, self.mean)
        yield 'step_002', x
        yield 'step_003', x
