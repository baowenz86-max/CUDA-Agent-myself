import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels: int, kernel_size: tuple, dilation: tuple, bias: bool, num_features: int):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA and float32
        # The custom op expects contiguous, but we don't call .contiguous() here; the binding will handle or require contiguous.
        # Since input from task contract is CPU but execution device is cuda, the harness moves it to cuda before forward.
        # We assume x is already on CUDA and contiguous.
        kernel_h, kernel_w = self.kernel_size
        dilation_h, dilation_w = self.dilation
        x = cuda_extension.op_unfold(x, kernel_h, kernel_w, dilation_h, dilation_w)
        # BatchNorm1d training mode: need weight, bias, running_mean, running_var, eps, momentum
        weight = self.bn.weight if self.bn.weight is not None else None
        bias = self.bn.bias if self.bn.bias is not None else None
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        eps = self.bn.eps
        momentum = self.bn.momentum
        x = cuda_extension.op_batch_norm_1d_training(x, weight, bias, running_mean, running_var, eps, momentum)
        return x

    def debug_forward(self, x: torch.Tensor):
        kernel_h, kernel_w = self.kernel_size
        dilation_h, dilation_w = self.dilation
        unfolded = cuda_extension.op_unfold(x, kernel_h, kernel_w, dilation_h, dilation_w)
        yield 'step_001', unfolded
        weight = self.bn.weight if self.bn.weight is not None else None
        bias = self.bn.bias if self.bn.bias is not None else None
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        eps = self.bn.eps
        momentum = self.bn.momentum
        output = cuda_extension.op_batch_norm_1d_training(unfolded, weight, bias, running_mean, running_var, eps, momentum)
        yield 'step_002', output
        yield 'step_003', output
