import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, num_classes, dim):
        super(Model, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=dim)

    def forward(self, x):
        x = self.max_pool(x)
        _, indices = torch.max(x, dim=1, keepdim=True)
        x = F.one_hot(indices, num_classes=num_classes)
        return x


batch_size = 16
height, width = 1024, 1024  # Increased spatial dimensions to increase computation
num_classes = 10
dim = 2


def get_inputs():
    return [torch.randn(batch_size, num_classes, height, width)]


def get_init_inputs():
    return [num_classes, num_classes, dim]