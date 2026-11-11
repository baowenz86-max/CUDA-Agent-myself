import torch


class Model(torch.nn.Module):
    def __init__(self, bin_edges):
        super(Model, self).__init__()
        self.bin_edges = bin_edges

    def forward(self, x):
        x = torch.mean(x, dim=(0, 2, 3))  # Apply torch.mean
        x = torch.histc(x, bins=len(self.bin_edges) - 1, min=0, max=1)  # Apply torch.histc
        return x


# Hyperparameters
batch_size = 512
channels = 128
height = 200
width = 200


# Define a function to provide inputs to the model
def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


# Define a function that returns the bin edges for the histogram
def get_init_inputs():
    return [torch.linspace(0, 1, 10)]  # Assuming the input feature has 10 elements