import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode, pools):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # 保持与参考 state_dict 一致：pools 为 ParameterList，但无参数
        self.pools = nn.ParameterList(pools)

    def forward(self, x):
        x = cuda_extension.interpolate_bilinear_scale(x, self.scale_factor)
        pooled_tensors = [
            cuda_extension.adaptive_avg_pool2d(x, pool.output_size[0], pool.output_size[1])
            for pool in self.pools
        ]
        return pooled_tensors

    def debug_forward(self, x):
        """Return generator yielding (checkpoint_id, value) in reference order."""
        x = cuda_extension.interpolate_bilinear_scale(x, self.scale_factor)
        yield "step_001", x

        pooled_tensors = [
            cuda_extension.adaptive_avg_pool2d(x, pool.output_size[0], pool.output_size[1])
            for pool in self.pools
        ]
        yield "step_002", pooled_tensors

        yield "step_003", pooled_tensors
