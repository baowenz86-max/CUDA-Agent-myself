import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_features, num_classes, dim, index):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x):
        x = F.log_softmax(x, dim=self.dim)
        x = torch.index_select(x, self.dim, self.index)
        return x


def get_inputs():
    batch_size = 32
    seq_len = 10000
    input_features = 256
    x = torch.randn(batch_size, seq_len, input_features)
    return [x]


def get_init_inputs():
    input_features = 1024
    num_classes = 20
    dim = 1
    index = torch.tensor([1, 2, 3, 4])
    return [input_features, num_classes, dim, index]