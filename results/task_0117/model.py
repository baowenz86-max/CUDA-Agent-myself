import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, num_classes):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=stride, padding=padding)
        self.num_classes = num_classes

    def forward(self, x, target):
        x = self.avg_pool(x)
        x = torch.tan(x)
        loss = nn.CrossEntropyLoss()(x.view(x.size(0), -1), target)
        return loss


batch_size = 32768
in_channels = 100
length = 19  # Ensures pooled length=10 when using kernel_size=3, stride=2, padding=1
kernel_size = 3
stride = 2
padding = 1
num_classes = 1000  # in_channels * pooled_length (100*10=1000)


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    target = torch.randint(0, num_classes, (batch_size,))
    return [x, target]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, num_classes]