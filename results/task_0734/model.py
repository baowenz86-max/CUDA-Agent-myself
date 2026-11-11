import torch


class Model(torch.nn.Module):
    def __init__(self, input_dim, squeeze_axes):
        super(Model, self).__init__()
        self.squeeze_axes = squeeze_axes

    def forward(self, x):
        x = torch.abs(x)
        x = torch.squeeze(x, dim=self.squeeze_axes)
        return x


def get_inputs():
    in_features = 3
    batch_size = 512
    input_dim = (batch_size, in_features, 1024, 1024)  # Increased spatial dimensions
    x = torch.randn(*input_dim)
    return [x]


def get_init_inputs():
    input_dim = get_inputs()[0].shape[1]
    squeeze_axes = 2
    return [input_dim, squeeze_axes]