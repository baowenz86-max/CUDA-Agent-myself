import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, num_params):
        super().__init__()
        self.linear = nn.Linear(in_features, num_params)
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_params)])

    def forward(self, input, target):
        x = self.linear(input)
        kldiv_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(x, dim=1), F.softmax(target, dim=1))
        l1_loss = sum(F.l1_loss(x[:, i], self.params[i].expand_as(x[:, i])) for i in range(len(self.params)))
        return kldiv_loss, l1_loss


batch_size = 128
in_features = 768  # This dimension (768) is already >=512, so the test input is not too small
num_params = 200  # Increased from 10 to increase computation


def get_inputs():
    input = torch.randn(batch_size, in_features)
    target = torch.randn(batch_size, num_params)
    return [input, target]


def get_init_inputs():
    return [in_features, num_params]