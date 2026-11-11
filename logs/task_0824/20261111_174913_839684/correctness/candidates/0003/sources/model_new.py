import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, num_features, kernel_size, stride, padding, dilation):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        x = cuda_extension.batch_norm2d_inference(
            x, self.batch_norm.weight, self.batch_norm.bias,
            self.batch_norm.running_mean, self.batch_norm.running_var, 1e-5
        )
        x = cuda_extension.permute_contiguous(x, torch.tensor([0, 2, 1, 3], dtype=torch.int64))
        x = cuda_extension.view(x, torch.tensor([x.size(0), x.size(1), -1], dtype=torch.int64))
        x = cuda_extension.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)
        x = cuda_extension.view(x, torch.tensor([x.size(0), x.size(1), -1, 1], dtype=torch.int64))
        x = cuda_extension.permute_contiguous(x, torch.tensor([0, 2, 1, 3], dtype=torch.int64))
        return x

    def debug_forward(self, x):
        x = cuda_extension.batch_norm2d_inference(
            x, self.batch_norm.weight, self.batch_norm.bias,
            self.batch_norm.running_mean, self.batch_norm.running_var, 1e-5
        )
        yield 'step_001', x
        x = cuda_extension.permute_contiguous(x, torch.tensor([0, 2, 1, 3], dtype=torch.int64))
        yield 'step_002', x
        x = cuda_extension.view(x, torch.tensor([x.size(0), x.size(1), -1], dtype=torch.int64))
        yield 'step_003', x
        x = cuda_extension.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)
        yield 'step_004', x
        x = cuda_extension.view(x, torch.tensor([x.size(0), x.size(1), -1, 1], dtype=torch.int64))
        yield 'step_005', x
        x = cuda_extension.permute_contiguous(x, torch.tensor([0, 2, 1, 3], dtype=torch.int64))
        yield 'step_006', x
        yield 'step_007', x
