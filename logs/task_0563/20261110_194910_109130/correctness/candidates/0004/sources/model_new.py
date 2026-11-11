import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, tensor_a, tensor_b):
        super().__init__()
        self.tensor_a = nn.Parameter(tensor_a)
        self.tensor_b = nn.Parameter(tensor_b)

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        equal_result = cuda_extension.tensor_equal(self.tensor_a, self.tensor_b)
        return x, equal_result

    def debug_forward(self, x):
        relu_x = cuda_extension.relu_forward(x)
        equal_result = cuda_extension.tensor_equal(self.tensor_a, self.tensor_b)
        yield "step_001", relu_x
        yield "step_002", equal_result
        yield "step_003", (relu_x, equal_result)
