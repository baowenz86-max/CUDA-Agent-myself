import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, y):
        x = torch.add(x, y)
        argmin = torch.argmin(x, dim=-1, keepdim=True)
        stft = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=True
        )
        return x, argmin, stft


batch_size = 128
audio_length = 200000  # Increased from 16000 to increase STFT computation
n_fft = 1024
hop_length = 256
win_length = 1024


def get_inputs():
    x = torch.randn(batch_size, audio_length)
    y = torch.randn(batch_size, audio_length)
    return [x, y]


def get_init_inputs():
    return [n_fft, hop_length, win_length]