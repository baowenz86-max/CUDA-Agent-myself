import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, sub_model_dict):
        super().__init__()
        self.model_dict = nn.ModuleDict(sub_model_dict)

    def forward(self, x):
        weights = []
        biases = []
        for key in self.model_dict.keys():
            sub_model = self.model_dict[key]
            weights.append(sub_model.linear.weight)
            biases.append(sub_model.linear.bias)
        return cuda_extension.batched_linear_forward(x, weights, biases)

    def debug_forward(self, *inputs):
        x = inputs[0]
        weights = []
        biases = []
        for key in self.model_dict.keys():
            sub_model = self.model_dict[key]
            weights.append(sub_model.linear.weight)
            biases.append(sub_model.linear.bias)
        out = cuda_extension.batched_linear_forward(x, weights, biases)
        yield 'step_001', out
