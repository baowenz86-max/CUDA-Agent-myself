import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.nll_loss = nn.NLLLoss()

    def forward(self, log_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.nll_loss(log_prob, target)


batch_size = 100000  # Increased batch size to increase computation time
num_classes = 10


def get_inputs():
    log_prob = torch.randn(batch_size, num_classes).log_softmax(dim=1)
    target = torch.empty(batch_size, dtype=torch.long).random_(num_classes)
    return [log_prob, target]


def get_init_inputs():
    return [10]