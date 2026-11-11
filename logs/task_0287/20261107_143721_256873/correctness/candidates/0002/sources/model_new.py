import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        # 保持与参考模型相同的 state_dict 键：batch_norm.*
        self.batch_norm = nn.BatchNorm2d(in_features)
        # 参考模型没有使用 out_features 和 bias 创建额外模块，但保留参数以示接口兼容
        # 注意：out_features 和 bias 参数不参与前向计算

    def forward(self, x):
        # step_001: batch_norm
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
        )
        # step_002: tanh
        x = cuda_extension.tanh_forward(x)
        # step_003: return x
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        # step_001: batch_norm
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
        )
        yield 'step_001', x
        # step_002: tanh
        x = cuda_extension.tanh_forward(x)
        yield 'step_002', x
        # step_003: return x
        yield 'step_003', x
