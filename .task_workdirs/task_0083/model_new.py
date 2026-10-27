import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode, pools):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # pools is a list of tuples, e.g. [(8,8), (4,4)]
        # store as plain list to match reference state_dict (no parameters)
        self.pools = pools

    def forward(self, x):
        x = cuda_extension.interpolate_bilinear_scale(x, self.scale_factor)
        pooled_tensors = [
            cuda_extension.adaptive_avg_pool2d(x, out_h, out_w)
            for out_h, out_w in self.pools
        ]
        return pooled_tensors

    def debug_forward(self, x):
        """Return generator yielding (checkpoint_id, value) in reference order."""
        x = cuda_extension.interpolate_bilinear_scale(x, self.scale_factor)
        yield "step_001", x

        pooled_tensors = [
            cuda_extension.adaptive_avg_pool2d(x, out_h, out_w)
            for out_h, out_w in self.pools
        ]
        yield "step_002", pooled_tensors

        yield "step_003", pooled_tensors
