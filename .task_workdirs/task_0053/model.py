import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, scaling_factor):
        super().__init__()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")
        self.scaling_factor = scaling_factor

    def forward(self, input, target, target_labels):
        kl_loss = self.kl_div_loss(F.log_softmax(input, dim=1), F.softmax(target, dim=1))
        ce_loss = self.cross_entropy_loss(input, target_labels)
        combined_loss = kl_loss + ce_loss
        remainder_loss = torch.remainder(combined_loss, self.scaling_factor)
        return remainder_loss


batch_size = 128
num_classes = 500  # Increased from 10 to increase computation
sequence_length = 512


def get_inputs():
    input = torch.randn(batch_size, num_classes, sequence_length)
    target = torch.randn(batch_size, num_classes, sequence_length)
    target_labels = torch.randint(0, num_classes, (batch_size, sequence_length))
    return [input, target, target_labels]


def get_init_inputs():
    return [num_classes, 10.0]