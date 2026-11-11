import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, n_mels):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels

    def forward(self, x):
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=True)
        x = torch.reshape(x, (x.size(0), x.size(1), -1))
        return x


batch_size = 128
n_fft = 512
hop_length = 256
win_length = 512
n_mels = 128
signal_length = 480000  # Increased from 16000 to increase runtime


def get_inputs():
    return [torch.randn(batch_size, signal_length)]


def get_init_inputs():
    return [n_fft, hop_length, win_length, n_mels]