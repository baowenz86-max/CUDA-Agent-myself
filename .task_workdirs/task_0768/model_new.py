import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        abs_x = cuda_extension.abs_forward(x)
        one_hot = cuda_extension.argmax_onehot_forward(abs_x, self.num_classes)
        return one_hot

    def debug_forward(self, x):
        abs_x = cuda_extension.abs_forward(x)
        yield 'step_001', abs_x
        one_hot = cuda_extension.argmax_onehot_forward(abs_x, self.num_classes)
        yield 'step_002', one_hot
        yield 'step_003', one_hot
