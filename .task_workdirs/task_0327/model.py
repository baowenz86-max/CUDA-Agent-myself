import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, n_mels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.conv1d = nn.Conv1d(n_fft // 2 + 1, n_mels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        x = torch.abs(x)
        x = torch.remainder(x, 1.0)
        x = self.conv1d(x)
        return x


batch_size = 128
n_fft = 512
hop_length = 256
n_mels = 128
kernel_size = 5
stride = 1
padding = 2
length = 160000  # Increased length to 160000 to increase runtime


def get_inputs():
    return [torch.randn(batch_size, length)]


def get_init_inputs():
    return [n_fft, hop_length, n_mels, kernel_size, stride, padding]