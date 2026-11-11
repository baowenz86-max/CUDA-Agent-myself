import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, elements, values):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.elements = nn.Parameter(torch.tensor(elements, dtype=torch.long), requires_grad=False)
        self.values = nn.Parameter(torch.tensor(values), requires_grad=False)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        mask = torch.isin(x, self.values)
        return x, mask


batch_size = 32
seq_length = 1024  # Increased from 512 to increase computation time
input_size = 16
hidden_size = 32
num_layers = 2
elements = torch.randint(0, 10, (batch_size, seq_length, input_size))
values = torch.tensor([3, 5, 7])


def get_inputs():
    return [torch.randn(batch_size, seq_length, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, num_layers, elements, values]