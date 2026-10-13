import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, bias_shape, kernel_size, stride, padding):
        super().__init__()
        # 保持与参考模型相同的模块和参数，确保 state_dict 键一致
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # 第一步：卷积（使用自定义 conv2d 算子，包含 conv.bias）
        x = torch.ops.cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias)
        # 第二步：加模型 bias
        x = torch.ops.cuda_extension.add_bias_forward(x, self.bias)
        # 第三步：BatchNorm eval 模式，使用 running stats
        x = torch.ops.cuda_extension.batch_norm_forward(
            x, self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var, self.bn.eps
        )
        # 第四步：平均池化
        x = torch.ops.cuda_extension.avg_pool2d_forward(
            x, kernel_size=self.pool.kernel_size, stride=self.pool.stride, padding=self.pool.padding
        )
        # 第五步：sigmoid
        x = torch.ops.cuda_extension.sigmoid_forward(x)
        return x

    def debug_forward(self, x):
        # 与 forward 相同的计算，但逐步 yield 中间结果
        x = torch.ops.cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias)
        yield "step_001", x
        x = torch.ops.cuda_extension.add_bias_forward(x, self.bias)
        yield "step_002", x
        x = torch.ops.cuda_extension.batch_norm_forward(
            x, self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var, self.bn.eps
        )
        yield "step_003", x
        x = torch.ops.cuda_extension.avg_pool2d_forward(
            x, kernel_size=self.pool.kernel_size, stride=self.pool.stride, padding=self.pool.padding
        )
        yield "step_004", x
        x = torch.ops.cuda_extension.sigmoid_forward(x)
        yield "step_005", x
        # 最终返回值也需要 yield，参考 checkpoint 中包含 return x
        yield "step_006", x
