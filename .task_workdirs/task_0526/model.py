import torch


class Model(torch.nn.Module):
    def __init__(self, input_size, subtract_value):
        super().__init__()
        self.subtract_value = subtract_value

    def forward(self, x):
        return x - self.subtract_value


# Hyperparameters determining input shape
batch_size = 64
seq_len = 4096
input_dim = 1250


def get_inputs():
    input_tensor = torch.randn(batch_size, seq_len, input_dim)
    return [input_tensor]


def get_init_inputs():
    subtract_value = torch.tensor(2.0)
    return [get_inputs()[0].shape[1], subtract_value]