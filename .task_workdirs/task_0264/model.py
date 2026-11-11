import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.log_softmax(x, dim=1)
        x = torch.argmax(x, dim=1)
        return x


def get_inputs():
    batch_size, seq_len, num_classes = 200, 1000, 1000  # Increased input dimensions
    x = torch.randn(batch_size, seq_len, num_classes)
    return [x]


def get_init_inputs():
    return []