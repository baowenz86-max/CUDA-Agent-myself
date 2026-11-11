import torch


class Model(torch.nn.Module):
    def __init__(self, cat_dim):
        super().__init__()
        self.cat_dim = cat_dim

    def forward(self, x):
        x = torch.unbind(x, dim=self.cat_dim)
        x = torch.cat(x, dim=0)
        return x


# Hyperparameters determining input shape
row_dim = 100000  # Increased from 512 to increase runtime
col_dim = 3


def get_inputs():
    x = torch.randn(row_dim, col_dim)  # Shape (100000, 3)
    return [x]


def get_init_inputs():
    return [0]  # The dimension along the input tensors will be concatenated