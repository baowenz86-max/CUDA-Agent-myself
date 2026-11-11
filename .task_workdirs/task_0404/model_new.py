import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, binary_threshold):
        super().__init__()
        self.binary_threshold = binary_threshold

    def forward(self, x):
        # 参考代码在 CUDA 设备上执行，确保输入位于 CUDA
        x = self._ensure_cuda(x)
        return cuda_extension.threshold_abs_forward(x, self.binary_threshold)

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = self._ensure_cuda(x)
        # 第一步：torch.abs(x)
        abs_result = cuda_extension.threshold_abs_forward(x, 0.0)  # 阈值 0.0 使 abs 完全保留
        yield ('step_001', abs_result)
        # 第二步：torch.where(...)
        thresholded_result = cuda_extension.threshold_abs_forward(x, self.binary_threshold)
        yield ('step_002', thresholded_result)
        # 第三步：return x
        yield ('step_003', thresholded_result)

    @staticmethod
    def _ensure_cuda(tensor):
        if tensor.device.type != 'cuda':
            tensor = tensor.cuda()
        return tensor

