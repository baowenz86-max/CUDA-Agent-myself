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
        x = cuda_extension.custom_reshape(x, (-1, x.size(-1)))
        x = cuda_extension.custom_stft(x, self.n_fft, self.hop_length, self.win_length, True)
        x = cuda_extension.custom_reshape(x, self.mel_shape)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_reshape(x, (-1, x.size(-1)))
        yield 'step_001', x
        x = cuda_extension.custom_stft(x, self.n_fft, self.hop_length, self.win_length, True)
        yield 'step_002', x
        x = cuda_extension.custom_reshape(x, self.mel_shape)
        yield 'step_003', x
        yield 'step_004', x
