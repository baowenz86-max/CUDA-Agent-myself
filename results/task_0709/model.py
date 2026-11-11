import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_images, num_bins, bins_range):
        super().__init__()
        self.num_images = num_images

    def forward(self, x):
        # Split the input tensor into a list of tensors along the batch dimension
        xs = torch.chunk(x, chunks=self.num_images, dim=0)

        # Apply torch.histc to each chunk independently and then concatenate them along the batch dimension
        ys = [torch.histc(chunk, bins=num_bins, min=bins_range[0], max=bins_range[1]) for chunk in xs]
        y = torch.cat(ys, dim=0)
        return y


# Define the input parameters
num_images = 2
num_bins = 10
bins_range = (0, 255)

# Define the data dimensions
batch_size = 512  # Increased to 512 to ensure at least one dimension >=512
channels = 3
height = 128  # Increased from 32 to 128 to increase computation
width = 128   # Increased from 32 to 128 to increase computation


# Define the get_inputs function
def get_inputs():
    # Create a batch of images with shape [batch_size, channels, height, width]
    return [torch.randn(batch_size, channels, height, width)]


# Define the get_init_inputs function
def get_init_inputs():
    # Return the parameters needed to initialize the model
    return [num_images, num_bins, bins_range]