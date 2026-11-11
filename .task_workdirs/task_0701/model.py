import torch


class Model(torch.nn.Module):
    def __init__(self, split_size, num_splits, axis):
        super().__init__()
        self.split_size = split_size
        self.num_splits = num_splits
        self.axis = axis

    def forward(self, x):
        # Split the tensor along the specified axis
        x = torch.split(x, self.split_size, self.axis)

        # Sum across the splits
        x = [torch.sum(t) for t in x]

        return x


# Define hyperparameters for input shape
batch_size = 512  # Increased to ensure at least one dimension >=512


# Define the inputs for the model
def get_inputs():
    # Create a dummy tensor with increased depth dimension to increase computation
    x = torch.randn(batch_size, 3, 384, 64)  # Changed 64 -> 384 in 3rd dimension
    return [x]


def get_init_inputs():
    # Define the parameters needed for the model initialization
    # Here we are splitting the 3rd dimension of the tensor into 2 parts of equal size
    return [2, 2, 2]