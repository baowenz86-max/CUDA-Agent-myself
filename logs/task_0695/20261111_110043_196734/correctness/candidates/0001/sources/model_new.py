import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, pad, dim):
        super().__init__()
        self.pad = tuple(pad)
        self.dim = dim

    def forward(self, x):
        # 输入可能在 CPU，但参考执行在 CUDA，需移动到 CUDA
        if not x.is_cuda:
            x = x.cuda()
        x = torch.ops.cuda_extension.custom_pad(x, self.pad[0], self.pad[1], self.pad[2], self.pad[3])
        x = torch.ops.cuda_extension.custom_cat(x, x, self.dim)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        padded = torch.ops.cuda_extension.custom_pad(x, self.pad[0], self.pad[1], self.pad[2], self.pad[3])
        yield ('step_001', padded)
        cat_out = torch.ops.cuda_extension.custom_cat(padded, padded, self.dim)
        yield ('step_002', cat_out)
        yield ('step_003', cat_out)
