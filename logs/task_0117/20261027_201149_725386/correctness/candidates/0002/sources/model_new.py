import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_classes = num_classes
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=stride, padding=padding)

    def forward(self, x, target):
        import cuda_extension
        x = cuda_extension.avgpool1d_forward(x, self.kernel_size, self.stride, self.padding)
        x = cuda_extension.tanh_forward(x)
        loss = cuda_extension.cross_entropy_loss_forward(x, target)
        return loss

    def debug_forward(self, *inputs):
        import cuda_extension
        x, target = inputs
        x = cuda_extension.avgpool1d_forward(x, self.kernel_size, self.stride, self.padding)
        yield 'step_001', x
        x = cuda_extension.tanh_forward(x)
        yield 'step_002', x
        loss = cuda_extension.cross_entropy_loss_forward(x, target)
        yield 'step_003', loss
        yield 'step_004', loss
