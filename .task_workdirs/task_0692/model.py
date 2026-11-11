import torch


class Model(torch.nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, dilation=1, use_selu=True, use_dig=True):
        super(Model, self).__init__()

        self.conv = torch.nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.use_selu = use_selu
        self.use_dig = use_dig

        if use_selu:
            self.act = torch.nn.SELU()
        else:
            self.act = torch.nn.Identity()

        if use_dig:
            self.dig = torch.nn.Identity()
        else:
            self.dig = torch.nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.dig(x)
        return x


batch_size = 512  # Increased batch size to 512 to meet the dimension requirement


def get_inputs():
    return [torch.randn(batch_size, 3, 128, 128)]  # Increased spatial dimensions to 128x128


def get_init_inputs():
    return [3, (3, 3), 1, 1, 1, True, True]