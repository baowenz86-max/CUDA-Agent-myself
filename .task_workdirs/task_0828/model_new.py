import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, n_mels: int, mel_shape: tuple):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.mel_shape = mel_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.custom_reshape_input(x)
        x = cuda_extension.custom_stft(x, self.n_fft, self.hop_length, self.win_length)
        x = cuda_extension.custom_reshape_output(x, 0)
        return x

    def debug_forward(self, x: torch.Tensor):
        x1 = cuda_extension.custom_reshape_input(x)
        yield 'step_001', x1
        x2 = cuda_extension.custom_stft(x1, self.n_fft, self.hop_length, self.win_length)
        yield 'step_002', x2
        x3 = cuda_extension.custom_reshape_output(x2, 0)
        yield 'step_003', x3
        yield 'step_004', x3
