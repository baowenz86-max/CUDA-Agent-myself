import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        # 参数命名与参考模型保持一致：instance_norm.weight / instance_norm.bias
        self.instance_norm = nn.Module()
        self.instance_norm.weight = nn.Parameter(torch.ones(num_features))
        self.instance_norm.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

    def forward(self, x):
        # 第一步：InstanceNorm2d(affine=True, track_running_stats=False)
        # 参考调用：self.instance_norm(x)，内部使用当前 weight 和 bias
        x = cuda_extension.instance_norm_forward(
            x, self.instance_norm.weight, self.instance_norm.bias, self.eps
        )
        # 第二步：SELU 激活函数
        x = cuda_extension.selu_forward(x)
        return x

    def debug_forward(self, x):
        # 按检查点顺序 yield (checkpoint_id, value)
        x = cuda_extension.instance_norm_forward(
            x, self.instance_norm.weight, self.instance_norm.bias, self.eps
        )
        yield 'step_001', x
        x = cuda_extension.selu_forward(x)
        yield 'step_002', x
        yield 'step_003', x
