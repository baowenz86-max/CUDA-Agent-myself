import torch


class Model(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, num_classes):
        super(Model, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.num_classes = num_classes

    def forward(self, x1, x2):
        gt_mask = torch.gt(x1, x2)
        concatenated = torch.cat((x1, x2), dim=1)
        return gt_mask, concatenated


batch_size = 1024  # Increased from 512 to increase runtime
input_dim1 = 128
input_dim2 = 128
num_classes = 1
height, width = 32, 32


def get_inputs():
    x1 = torch.randn(batch_size, input_dim1, height, width)
    x2 = torch.randn(batch_size, input_dim2, height, width)
    return [x1, x2]


def get_init_inputs():
    return [input_dim1, input_dim2, num_classes]