import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_dim1, input_dim2, num_classes):
        super(ModelNew, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.num_classes = num_classes

    def forward(self, x1, x2):
        gt_mask = cuda_extension.greater_than(x1, x2)
        concatenated = cuda_extension.concatenate_dim1(x1, x2)
        return (gt_mask, concatenated)

    def debug_forward(self, x1, x2):
        gt_mask = cuda_extension.greater_than(x1, x2)
        yield ("step_001", gt_mask)
        concatenated = cuda_extension.concatenate_dim1(x1, x2)
        yield ("step_002", concatenated)
        yield ("step_003", (gt_mask, concatenated))
