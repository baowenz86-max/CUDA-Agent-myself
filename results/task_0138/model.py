import torch


class Model(torch.nn.Module):
    def __init__(self, size, p, dim):
        super(Model, self).__init__()
        self.diag = torch.diag(torch.randn(size))
        self.norm = torch.norm(self.diag, p, dim)

    def forward(self, x):
        x = torch.diag_embed(x)
        x = torch.norm(x, p=1, dim=1)
        return x


input_dim = 512


def get_inputs():
    return [torch.randn(input_dim, 1500)]


def get_init_inputs():
    return [10, 2, -2]