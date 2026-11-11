import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, tensor1, tensor2):
        cat_out = cuda_extension.cat_forward(tensor1, tensor2)
        cos_out = cuda_extension.cos_forward(cat_out)
        return cos_out

    def debug_forward(self, tensor1, tensor2):
        cat_out = cuda_extension.cat_forward(tensor1, tensor2)
        yield ('step_001', cat_out)
        cos_out = cuda_extension.cos_forward(cat_out)
        yield ('step_002', cos_out)
        yield ('step_003', cos_out)
