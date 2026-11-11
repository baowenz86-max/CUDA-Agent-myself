import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, logits, targets, input_lengths, target_lengths):
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction="mean")
        return loss


batch_size = 16
input_dim = 128
num_classes = 30
seq_length = 2048  # Increased from 512 to 2048 to increase runtime
target_seq_length = 30


def get_inputs():
    logits = torch.randn(seq_length, batch_size, num_classes).log_softmax(2).detach().requires_grad_()
    targets = torch.randint(1, num_classes - 1, (batch_size, target_seq_length), dtype=torch.int)
    input_lengths = torch.full((batch_size,), seq_length, dtype=torch.int)
    target_lengths = torch.randint(10, target_seq_length, (batch_size,), dtype=torch.int)
    return [logits, targets, input_lengths, target_lengths]


def get_init_inputs():
    return [input_dim, num_classes]