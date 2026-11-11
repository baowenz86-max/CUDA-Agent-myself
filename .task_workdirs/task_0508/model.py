import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        stacked_predictions = torch.stack(y_pred)
        stacked_true_labels = torch.stack(y_true)
        return self.loss(stacked_predictions, stacked_true_labels)


# Hyperparameters with at least one dimension >= 512
num_samples = 100
batch_size = 512  # Increased batch size to 512
channels = 1
height = 100  # Increased from 10 to 100
width = 100   # Increased from 10 to 100


def get_inputs():
    y_pred = [torch.randn(batch_size, channels, height, width) for _ in range(num_samples)]
    y_true = [torch.rand(batch_size, channels, height, width).round() for _ in range(num_samples)]
    return y_pred, y_true


def get_init_inputs():
    # Since the initialization involves no parameters, we return an empty list
    return []