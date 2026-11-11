import torch


class Model(torch.nn.Module):
    def __init__(self, num_features):
        super(Model, self).__init__()
        self.batchnorm1d = torch.nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.batchnorm1d(x)


def get_inputs():
    x = torch.randn(16, 64, 128000)  # Increased sequence length to 128000
    return [x]


def get_init_inputs():
    return [64]  # num_features remains 64