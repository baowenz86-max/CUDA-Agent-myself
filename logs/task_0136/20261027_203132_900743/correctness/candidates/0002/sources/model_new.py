import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.fc = nn.Linear(input_features, 10)

    def forward(self, x):
        x = cuda_extension.sigmoid_forward(x)
        x = cuda_extension.avg_pool2d_forward(x, kernel_size=2, stride=2, padding=0)
        x = cuda_extension.flatten_forward(x, start_dim=1)
        x = cuda_extension.linear_forward(x, self.fc.weight, self.fc.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.sigmoid_forward(x)
        yield 'step_001', x
        x = cuda_extension.avg_pool2d_forward(x, kernel_size=2, stride=2, padding=0)
        yield 'step_002', x
        x = cuda_extension.flatten_forward(x, start_dim=1)
        yield 'step_003', x
        x = cuda_extension.linear_forward(x, self.fc.weight, self.fc.bias)
        yield 'step_004', x
        yield 'step_005', x


def get_inputs():
    batch_size, channels, height, width = 512, 3, 256, 256
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    input_features = 3 * 128 * 128
    return [input_features]
