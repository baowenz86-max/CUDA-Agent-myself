import torch


class Model(torch.nn.Module):
    def __init__(self, in_features):
        super(Model, self).__init__()
        self.in_features = in_features

    def forward(self, x):
        x, _ = torch.max(x, dim=1)
        x = torch.round(x)
        return x


batch_size = 512
dim2 = 32
dim3 = 256
dim4 = 256


def get_inputs():
    return [torch.randn(batch_size, dim2, dim3, dim4)]


def get_init_inputs():
    return [32]