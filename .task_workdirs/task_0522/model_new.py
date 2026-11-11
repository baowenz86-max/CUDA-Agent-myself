import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        # 参考实现中的 torch.equal(x, x) 结果不影响输出，可以不计算
        # 但为了 debug_forward 检查点，保留实际计算
        # 这里不调用 torch.equal 以避免进入自定义算子范围，正常 forward 只需返回结果
        x = cuda_extension.batch_norm_eval(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        return x

    def debug_forward(self, x):
        # 按检查点顺序 yield，包括最终返回
        yield 'step_001', True  # torch.equal(x, x) 恒为 True
        y = cuda_extension.batch_norm_eval(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield 'step_002', y
        yield 'step_003', y
