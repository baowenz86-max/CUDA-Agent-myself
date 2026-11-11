import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # 不创建任何 Parameter 或 Module，保持 state_dict 为空，与参考模型一致。

    def forward(self, x):
        x = cuda_extension.softplus_forward(x)
        x = cuda_extension.instance_norm3d_forward(x, eps=1e-05)
        return x

    def debug_forward(self, x):
        x = cuda_extension.softplus_forward(x)
        yield ("step_001", x)
        x = cuda_extension.instance_norm3d_forward(x, eps=1e-05)
        yield ("step_002", x)
        yield ("step_003", x)
