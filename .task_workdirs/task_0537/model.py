import torch


class Model(torch.nn.Module):
    def __init__(self, threshold, bucket_boundaries):
        super(Model, self).__init__()
        self.threshold = threshold
        self.bucket_boundaries = bucket_boundaries

    def forward(self, x):
        x = torch.where(x > self.threshold, x, torch.tensor(0.0))
        x = torch.round(x)
        x = torch.bucketize(x, self.bucket_boundaries)
        return x


input_dim1 = 8192
input_dim2 = 8192


def get_inputs():
    x = torch.randn(input_dim1, input_dim2)
    return [x]


def get_init_inputs():
    threshold = 0.5
    bucket_boundaries = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    return [threshold, bucket_boundaries]