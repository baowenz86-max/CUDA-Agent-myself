import torch


class Model(torch.nn.Module):
    def __init__(self, tensor1, tensor2):
        super(Model, self).__init__()
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def forward(self, mask):
        x = torch.sub(self.tensor1, self.tensor2)
        x = torch.masked_fill(x, mask, 0)
        return x


tensor_shape = (512, 512, 512)  # At least one dimension >= 512


def get_inputs():
    mask = torch.randint(0, 2, tensor_shape, dtype=torch.bool)
    return [mask]


def get_init_inputs():
    tensor1 = torch.randn(tensor_shape)
    tensor2 = torch.randn(tensor_shape)
    return [tensor1, tensor2]