import torch


class Model(torch.nn.Module):
    def __init__(self, input_dim, cumprod_dim, cumsum_dim, matmul_dims):
        super().__init__()
        self.input_dim = input_dim
        self.cumprod_dim = cumprod_dim
        self.cumsum_dim = cumsum_dim
        self.matmul_dims = matmul_dims

    def forward(self, x):
        # Cumulative product operation
        x = torch.cumprod(x, self.cumprod_dim)
        # Cumulative sum operation
        x = torch.cumsum(x, self.cumsum_dim)
        # Matrix multiplication operation
        x = torch.matmul(x, self.matmul_dims[0])
        x = torch.matmul(x, self.matmul_dims[1])
        return x


def get_inputs():
    batch_size = 64
    seq_length = 1024
    hidden_size = 512
    x = torch.rand(batch_size, seq_length, hidden_size)
    return [x]


def get_init_inputs():
    input_dim = 512
    cumprod_dim = 1
    cumsum_dim = 1
    matmul_dims = (torch.rand(512, 256), torch.rand(256, 128))
    return [input_dim, cumprod_dim, cumsum_dim, matmul_dims]