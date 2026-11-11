import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, pool_size):
        super().__init__()
        # 参考模型不注册任何参数，state_dict 为空
        self.pool_size = pool_size

    def forward(self, x):
        x = cuda_extension.custom_abs_forward(x)
        x = cuda_extension.adaptive_avg_pool2d_forward(
            x, self.pool_size[0], self.pool_size[1]
        )
        return x

    def debug_forward(self, x):
        # 按 task_contract 检查点顺序 yield (id, value)
        x = cuda_extension.custom_abs_forward(x)
        yield "step_001", x
        x = cuda_extension.adaptive_avg_pool2d_forward(
            x, self.pool_size[0], self.pool_size[1]
        )
        yield "step_002", x
        yield "step_003", x
