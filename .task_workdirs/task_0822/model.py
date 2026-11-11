import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value, head_bias, relu_input_shape):
        super().__init__()
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value, head_bias)
        self.relu_input_shape = relu_input_shape

    def forward(self, x, target=None, relu_input=None):
        if relu_input is None:
            relu_input = torch.randn(self.relu_input_shape)
        relu_output = torch.relu(relu_input)
        if target is not None:
            return self.adaptive_log_softmax(x, target), relu_output
        else:
            return self.adaptive_log_softmax.log_prob(x), relu_output


in_features = 768
n_classes = 50000
cutoffs = [10000, 40000]
div_value = 4.0
head_bias = False
batch_size = 1024  # Increased from 512 to 1024 to increase runtime
relu_input_shape = (batch_size, 128)


def get_inputs():
    x = torch.randn(batch_size, in_features)
    target = torch.randint(0, n_classes, (batch_size,))
    relu_input = torch.randn(relu_input_shape)
    return [x, target, relu_input]


def get_init_inputs():
    return [in_features, n_classes, cutoffs, div_value, head_bias, relu_input_shape]