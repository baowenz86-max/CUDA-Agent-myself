import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, cat_dim, argmax_dim):
        super().__init__()
        self.input_size = input_size
        self.cat_dim = cat_dim
        self.argmax_dim = argmax_dim

    def forward(self, x):
        # Split the input tensor into two halves
        half_size = self.input_size // 2
        x1, x2 = x[:, :, :half_size], x[:, :, half_size:]

        # Concatenate along the specified dimension
        cat_result = torch.cat((x1, x2), dim=self.cat_dim)

        # Find the indices of the maximum values along the specified dimension
        argmax_result = torch.argmax(x, dim=self.argmax_dim)

        return cat_result, argmax_result


def get_inputs():
    batch_size = 128
    seq_len = 2000
    input_size = 512
    x = torch.randn(batch_size, seq_len, input_size)
    return [x]


def get_init_inputs():
    x = get_inputs()[0]
    return [x.size(2), 1, 2]