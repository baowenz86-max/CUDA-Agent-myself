import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, n_mels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.conv1d = nn.Conv1d(n_fft // 2 + 1, n_mels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.custom_stft(x, self.n_fft, self.hop_length)
        x = cuda_extension.custom_abs(x)
        x = cuda_extension.custom_remainder(x, 1.0)
        x = cuda_extension.custom_conv1d(x, self.conv1d.weight, self.conv1d.bias, self.conv1d.stride[0], self.conv1d.padding[0])
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.custom_stft(x, self.n_fft, self.hop_length)
        yield 'step_001', x
        x = cuda_extension.custom_abs(x)
        yield 'step_002', x
        x = cuda_extension.custom_remainder(x, 1.0)
        yield 'step_003', x
        x = cuda_extension.custom_conv1d(x, self.conv1d.weight, self.conv1d.bias, self.conv1d.stride[0], self.conv1d.padding[0])
        yield 'step_004', x
        yield 'step_005', x
