import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, reduction_indices):
        super().__init__()
        self.input_size = input_size
        self.reduction_indices = reduction_indices

    def forward(self, x):
        x = cuda_extension.copy_tensor_k_abs_out(x)
        x = cuda_extension.copy_tensor_k_argmin_out(x, self.reduction_indices)
        return x

    def debug_forward(self, x):
        abs_result = cuda_extension.copy_tensor_k_abs_out(x)
        yield 'step_001', abs_result
        argmin_result = cuda_extension.copy_tensor_k_argmin_out(abs_result, self.reduction_indices)
        yield 'step_002', argmin_result
        yield 'step_003', argmin_result
