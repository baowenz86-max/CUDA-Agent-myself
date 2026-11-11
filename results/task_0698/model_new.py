import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # 参考模型没有可学习参数，state_dict 应为空

    def forward(self, x):
        # 输入可能位于 CPU（验证时 task_contract 中 device 为 cpu），但执行需要 CUDA
        # 自定义算子要求输入为 CUDA tensor，因此如果非 CUDA 则先转移到 cuda
        if not x.is_cuda:
            x = x.to('cuda')
        x = cuda_extension.mean_dim1_keepdim(x)
        x = cuda_extension.relu(x)
        return x

    def debug_forward(self, *inputs):
        if len(inputs) != 1:
            raise ValueError('debug_forward expects exactly one input tensor')
        x = inputs[0]
        if not x.is_cuda:
            x = x.to('cuda')
        x = cuda_extension.mean_dim1_keepdim(x)
        yield 'step_001', x
        x = cuda_extension.relu(x)
        yield 'step_002', x
        yield 'step_003', x
