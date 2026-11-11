import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # 参考模型没有参数，state_dict 应当为空

    def forward(self, x):
        x = x.to(device='cuda')
        x = cuda_extension.sigmoid_forward(x)
        x = cuda_extension.hardtanh_forward(x)
        return x

    def debug_forward(self, x):
        x = x.to(device='cuda')
        sig = cuda_extension.sigmoid_forward(x)
        yield 'step_001', sig
        hard = cuda_extension.hardtanh_forward(sig)
        yield 'step_002', hard
        yield 'step_003', hard
