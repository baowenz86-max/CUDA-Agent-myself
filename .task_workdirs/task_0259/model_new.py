import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, n_mels):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels

    def forward(self, x):
        x = cuda_extension.stft_forward(x, self.n_fft, self.hop_length, self.win_length)
        x = cuda_extension.reshape_forward(x, x.shape[0], x.shape[1])
        return x

    def debug_forward(self, x):
        x = cuda_extension.stft_forward(x, self.n_fft, self.hop_length, self.win_length)
        yield 'step_001', x
        x = cuda_extension.reshape_forward(x, x.shape[0], x.shape[1])
        yield 'step_002', x
        yield 'step_003', x
