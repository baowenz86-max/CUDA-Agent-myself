import torch


class Model(torch.nn.Module):
    def __init__(self, num_features, bin_boundaries):
        super(Model, self).__init__()
        self.num_features = num_features
        self.bin_boundaries = bin_boundaries

    def forward(self, x):
        x = torch.amax(x, dim=1, keepdim=True)
        x = torch.bucketize(x, self.bin_boundaries)
        x = torch.prod(x, dim=1)
        return x


def get_inputs():
    return [torch.randn(64, 20000, 768)]


def get_init_inputs():
    num_features = 512
    bin_boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    return [num_features, bin_boundaries]