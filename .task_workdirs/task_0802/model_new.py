import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_classes, num_prelu_features, prelu_alpha_value):
        super().__init__()
        # 保持与参考模型相同的 state_dict 键
        self.prelu = nn.PReLU(num_parameters=num_prelu_features, init=prelu_alpha_value)

    def forward(self, x):
        x = cuda_extension.custom_softmax_channel_only(x)
        x = cuda_extension.custom_prelu(x, self.prelu.weight)
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.custom_softmax_channel_only(x)
        yield 'step_001', x1
        x2 = cuda_extension.custom_prelu(x1, self.prelu.weight)
        yield 'step_002', x2
        yield 'step_003', x2
