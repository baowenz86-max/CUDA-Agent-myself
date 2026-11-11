import torch


class Model(torch.nn.Module):
    def __init__(self, divisor, modulus):
        super().__init__()
        self.divisor = divisor
        self.modulus = modulus

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.fmod(x, self.divisor)
        x = torch.fmod(x, self.modulus)
        return x


batch_size = 512
channels = 3
height = 512
width = 512


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [5, 3]