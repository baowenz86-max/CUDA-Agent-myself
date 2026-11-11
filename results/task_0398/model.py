import torch


# Define the Model
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        x = torch.softmax(x, dim=self.input_size)
        x = torch.sub(x, self.output_size)
        return x


# Prepare Inputs
batch_size = 512
height, width = 2048, 64  # Increased height to 2048 to increase computation
input_size = 1
output_size = 0.5


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return [input_size, output_size]