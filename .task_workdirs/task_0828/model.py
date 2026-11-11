import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, n_mels: int, mel_shape: tuple):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.mel_shape = mel_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.reshape(x, (-1, x.size(-1)))
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=True)
        x = torch.reshape(x, self.mel_shape)
        return x


batch_size = 1024  # Increased from 16 to increase computational load
num_samples = 16384  # Kept as is since it's already large
n_fft = 4096  # Increased from 2048 to make STFT more computationally intensive
hop_length = 512  # Kept constant
win_length = 4096  # Increased from 2048 to match new n_fft
n_mels = 128  # Kept constant
mel_shape = (batch_size, n_fft // 2 + 1, -1)  # Updated to match new n_fft


def get_inputs():
    return [torch.randn(batch_size, num_samples)]


def get_init_inputs():
    return [n_fft, hop_length, win_length, n_mels, mel_shape]