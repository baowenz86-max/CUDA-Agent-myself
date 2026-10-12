import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, bucket_boundaries_shape):
        super().__init__()
        self.num_classes = num_classes
        self.bucket_boundaries = nn.Parameter(torch.randn(bucket_boundaries_shape))

    def forward(self, x, values):
        x = F.one_hot(x, num_classes=self.num_classes)
        x = torch.maximum(x, torch.tensor(0.0))
        x = torch.bucketize(values, self.bucket_boundaries)
        return x


batch_size = 1024
sequence_length = 2000
num_classes = 128
bucket_boundaries_shape = (1000,)


def get_inputs():
    x = torch.randint(0, num_classes, (batch_size, sequence_length))
    values = torch.randn(batch_size, sequence_length)
    return [x, values]


def get_init_inputs():
    return [num_classes, bucket_boundaries_shape]