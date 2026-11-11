import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, padding):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.reflection_pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        x = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=True
        ).abs()
        x = self.reflection_pad(x)
        return x


batch_size = 128
length = 160000  # Increased input length to increase runtime
n_fft = 512
hop_length = 256
win_length = 512
padding = (5, 5, 0, 0)


def get_inputs():
    return [torch.randn(batch_size, length)]


def get_init_inputs():
    return [n_fft, hop_length, win_length, padding]