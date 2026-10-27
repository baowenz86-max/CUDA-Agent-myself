import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features, dim, index):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dim = dim
        self.index = index

    def forward(self, x):
        x = self.linear(x)
        x = torch.index_select(x, self.dim, self.index)
        x = self.relu(x)
        x = torch.transpose(x, self.dim, self.dim ^ 1)
        return x


batch_size = 512
in_features = 128
out_features = 256
dim = 1
height = 512  # Increased from 32 to 512 to increase computation


def get_inputs():
    x = torch.randn(batch_size, height, in_features)
    return [x]


def get_init_inputs():
    index = torch.tensor(range(0, height, 2))  # Index length increases with height
    return [in_features, out_features, dim, index]