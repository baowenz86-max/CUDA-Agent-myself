import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        max_val_1, _ = cuda_extension.max_dim1_keepdim(x)
        max_val_2, _ = cuda_extension.max_dim2_no_keepdim(max_val_1)
        return max_val_2

    def debug_forward(self, x):
        max_val_1, indices_1 = cuda_extension.max_dim1_keepdim(x)
        yield "step_001", (max_val_1, indices_1)
        max_val_2, indices_2 = cuda_extension.max_dim2_no_keepdim(max_val_1)
        yield "step_002", (max_val_2, indices_2)
        yield "step_003", max_val_2
