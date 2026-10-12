import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, clip_value, bucket_boundaries, bmm_dim):
        super().__init__()
        self.clip_value = clip_value
        self.bucket_boundaries = bucket_boundaries
        self.bmm_dim = bmm_dim

    def forward(self, x, y, z):
        x = torch.clip(x, max=self.clip_value)
        y = torch.bucketize(y, self.bucket_boundaries)
        z = torch.bmm(z, z.transpose(1, 2))
        return x, y, z


# Define test data
clip_value = 0.5
bucket_boundaries = torch.tensor([0, 0.25, 0.5, 0.75, 1])
batch_size = 128  # Increased from 64 to 128 to increase runtime
tensor_dim = 512  # Maintained at 512 as per requirement
x = torch.randn(batch_size, tensor_dim)
y = torch.rand(batch_size)
z = torch.randn(batch_size, tensor_dim, tensor_dim)


def get_inputs():
    return [x, y, z]


def get_init_inputs():
    return [clip_value, bucket_boundaries, tensor_dim]