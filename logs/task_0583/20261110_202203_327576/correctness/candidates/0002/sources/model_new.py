import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        # 参考模型没有可训练参数，保持 state_dict 为空
        self.input_channels = input_channels
        self.num_classes = num_classes

    def forward(self, x):
        # 检测 inf 元素
        inf_mask = cuda_extension.isinf_like(x)
        # 将 inf 替换为 -inf
        x = cuda_extension.where_inf_replace(inf_mask, x)
        # 沿 dim=1 执行 log_softmax
        x = cuda_extension.logsoftmax_dim1(x)
        return x

    def debug_forward(self, *inputs):
        (x,) = inputs
        inf_mask = cuda_extension.isinf_like(x)
        yield "step_001", inf_mask
        x = cuda_extension.where_inf_replace(inf_mask, x)
        yield "step_002", x
        x = cuda_extension.logsoftmax_dim1(x)
        yield "step_003", x
        yield "step_004", x
