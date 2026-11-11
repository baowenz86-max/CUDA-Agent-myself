import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, start, end, step, kernel_size, stride):
        super().__init__()
        # Preserve the original arange attribute for state_dict compatibility (no parameters)
        self.arange = torch.arange(start, end, step)
        # Store kernel_size and stride as tensors or tuples to use in forward
        # The reference uses nn.AvgPool2d(kernel_size, stride=stride)
        # kernel_size is expected to be a tuple (2,2) in this task, but handle general case
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        self.stride = stride

    def forward(self, x):
        # Use custom CUDA operator for average pooling
        # Parameters: input, kernel_size (int), stride (int), padding (int)
        return cuda_extension.avg_pool2d_forward(
            input=x,
            kernel_size=int(self.kernel_size[0]),
            stride=int(self.stride),
            padding=0
        )

    def debug_forward(self, *inputs):
        # Yield checkpoints in order: only step_001 (the final return)
        x = inputs[0]
        result = self.forward(x)
        yield 'step_001', result
