import torch
import torch.nn as nn


class SubModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class Model(nn.Module):
    def __init__(self, sub_model_dict):
        super().__init__()
        self.model_dict = nn.ModuleDict(sub_model_dict)

    def forward(self, x):
        return torch.stack([self.model_dict[key](x) for key in self.model_dict.keys()])


batch_size = 512
in_features = 128
out_features = 64
num_models = 200  # Increased number of sub-models to increase computation


def get_inputs():
    x = torch.randn(batch_size, in_features)
    return [x]


def get_init_inputs():
    sub_model_dict = {f"sub_model_{i}": SubModel(in_features, out_features) for i in range(num_models)}
    return [sub_model_dict]