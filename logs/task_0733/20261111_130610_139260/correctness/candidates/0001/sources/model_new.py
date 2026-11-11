import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, divisor, modulus):
        super().__init__()
        self.divisor = divisor
        self.modulus = modulus

    def forward(self, x):
        x = cuda_extension.mean_channel_fmod(x, float(self.divisor))
        x = cuda_extension.fmod_scalar(x, float(self.modulus))
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.mean_channel_fmod(x, float(self.divisor))
        yield ("step_001", x1)
        yield ("step_002", x1)
        x2 = cuda_extension.fmod_scalar(x1, float(self.modulus))
        yield ("step_003", x2)
        yield ("step_004", x2)
