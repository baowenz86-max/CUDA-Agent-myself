import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = cuda_extension.custom_upsample_bilinear2d(x, float(self.upsample.scale_factor), self.upsample.align_corners)
        x = cuda_extension.custom_mean_dim1_keepdim(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_upsample_bilinear2d(x, float(self.upsample.scale_factor), self.upsample.align_corners)
        yield ("step_001", x)
        x = cuda_extension.custom_mean_dim1_keepdim(x)
        yield ("step_002", x)
        yield ("step_003", x)
