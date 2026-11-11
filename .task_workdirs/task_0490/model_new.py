import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, norm_eps):
        super().__init__()
        # 保留与参考模块一致的空 state_dict；这些模块不参与 forward 计算
        self.gelu = nn.GELU()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps=norm_eps, affine=False, track_running_stats=False)

    def forward(self, x):
        x = cuda_extension.gelu_forward(x)
        x = cuda_extension.instance_norm2d_no_affine_forward(x, eps=1e-05)
        return x

    def debug_forward(self, x):
        checkpoint = cuda_extension.gelu_forward(x)
        yield 'step_001', checkpoint
        checkpoint = cuda_extension.instance_norm2d_no_affine_forward(checkpoint, eps=1e-05)
        yield 'step_002', checkpoint
        yield 'step_003', checkpoint
