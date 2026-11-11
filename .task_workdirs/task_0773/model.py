import torch


class Model(torch.nn.Module):
    def __init__(self, dimension):
        super(Model, self).__init__()
        self.dimension = dimension

    def forward(self, x, y):
        stacked_tensors = torch.stack([x, y], dim=self.dimension)
        result = torch.all(stacked_tensors, dim=self.dimension)
        return result


def get_inputs():
    batch_size = 128  # Increased from 16 to 128 to increase runtime
    channels = 3
    height = 512
    width = 512
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)
    return [x, y]


def get_init_inputs():
    return [0]  # For dimension 0 (stack along first dimension)