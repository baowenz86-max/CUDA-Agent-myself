import torch


class Model(torch.nn.Module):
    def __init__(self, dim, soft_max_dim):
        super().__init__()
        self.soft_max_dim = soft_max_dim
        self.add_value = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x):
        x = torch.softmax(x, dim=self.soft_max_dim)
        x = x + self.add_value
        return x


batch_size = 64
seq_len = 1024
vector_dim = 2048


def get_inputs():
    return [torch.randn(batch_size, seq_len, vector_dim)]


def get_init_inputs():
    return [vector_dim, 2]