import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # 保持与参考模型相同的 state_dict（此处为空）

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 参考实现: return self.loss(input, target)
        # 输入来自 CPU，计算在 CUDA 上执行，因此先转移到 CUDA。
        input_cuda = input.cuda()
        target_cuda = target.cuda()
        return cuda_extension.l1_loss_forward(input_cuda, target_cuda)

    def debug_forward(self, *inputs):
        # 按 checkpoints 顺序 yield (id, value)，包括最终返回
        input_tensor, target_tensor = inputs
        input_cuda = input_tensor.cuda()
        target_cuda = target_tensor.cuda()
        result = cuda_extension.l1_loss_forward(input_cuda, target_cuda)
        yield 'step_001', result
