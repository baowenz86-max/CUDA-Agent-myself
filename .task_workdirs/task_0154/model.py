import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        n_fft,
        win_length,
        hop_length,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        index_dim,
        indices,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.index_dim = index_dim
        self.indices = nn.Parameter(torch.tensor(indices), requires_grad=False)

    def forward(self, x):
        x = torch.stft(x, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, return_complex=True)
        x = torch.view_as_real(x).permute(0, 3, 1, 2)  # Shape (batch, 2, freq, frame)
        x = self.conv2d(x)
        x = torch.index_select(x, self.index_dim, self.indices)
        return x


batch_size = 32
n_fft = 512
win_length = 512
hop_length = 256
in_channels = 2  # Real and imaginary parts
out_channels = 32
kernel_size = (3, 3)
stride = 1
padding = (1, 1)
dilation = (1, 1)
bias = True
index_dim = 2
indices = [0, 1, 2, 3, 4]  # Example indices for frequency dimension
sequence_length = 81920  # Increased from 8192 to increase runtime


def get_inputs():
    x = torch.randn(batch_size, sequence_length)
    return [x]


def get_init_inputs():
    return [
        n_fft,
        win_length,
        hop_length,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        index_dim,
        indices,
    ]