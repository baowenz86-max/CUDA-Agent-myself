import torch


class Model(torch.nn.Module):
    def __init__(self, input_size, dim, index):
        super(Model, self).__init__()
        self.input_size = input_size
        self.dim = dim
        self.index = index

    def forward(self, x):
        x = torch.index_select(x, self.dim, self.index)
        return torch.minimum(x, torch.tensor(self.input_size))


input_dim1 = 100000
input_dim2 = 1000


def get_inputs():
    return [torch.randn(input_dim1, input_dim2)]


def get_init_inputs():
    return [input_dim1, 0, torch.arange(input_dim1)]