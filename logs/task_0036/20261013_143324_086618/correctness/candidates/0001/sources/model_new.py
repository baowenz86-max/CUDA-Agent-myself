import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, y):
        x = cuda_extension.custom_add(x, y)
        argmin = cuda_extension.custom_argmin(x)
        stft = cuda_extension.custom_stft(
            x, self.n_fft, self.hop_length, self.win_length, True
        )
        return x, argmin, stft

    def debug_forward(self, *inputs):
        x, y = inputs
        x = cuda_extension.custom_add(x, y)
        yield 'step_001', x
        argmin = cuda_extension.custom_argmin(x)
        yield 'step_002', argmin
        stft = cuda_extension.custom_stft(
            x, self.n_fft, self.hop_length, self.win_length, True
        )
        yield 'step_003', stft
        yield 'step_004', (x, argmin, stft)
