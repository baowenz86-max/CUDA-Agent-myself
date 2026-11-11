import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, input_dim, max_threshold, any_threshold):
        super().__init__()
        self.max_threshold = max_threshold
        self.any_threshold = any_threshold

    def forward(self, x):
        return cuda_extension.custom_threshold_mask(x, self.max_threshold, self.any_threshold)

    def debug_forward(self, x):
        max_result, _ = cuda_extension.custom_max_dim(x, 1, True)
        yield 'step_001', (max_result, _)
        max_mask = cuda_extension.custom_gt_scalar(max_result, self.max_threshold)
        yield 'step_002', max_mask
        any_mask = cuda_extension.custom_any_gt(x, self.any_threshold, 1, True)
        yield 'step_003', any_mask
        combined_mask = cuda_extension.custom_logical_and(max_mask, any_mask)
        yield 'step_004', combined_mask
        result = cuda_extension.custom_where_mask(combined_mask, x)
        yield 'step_005', result
        yield 'step_006', result
