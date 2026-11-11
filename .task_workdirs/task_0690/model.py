import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensor1, tensor2):
        output = torch.cat((tensor1, tensor2), dim=1)
        output = torch.cos(output)
        return output


batch_size = 512
channels = 8
height = 256
width = 256


def get_inputs():
    tensor1 = torch.randn(batch_size, channels, height, width)
    tensor2 = torch.randn(batch_size, channels, height, width)
    return [tensor1, tensor2]


def get_init_inputs():
    return []