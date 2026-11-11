import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value, head_bias, relu_input_shape):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs
        self.div_value = div_value
        self.head_bias = head_bias
        self.relu_input_shape = relu_input_shape

        # Define parameters to match AdaptiveLogSoftmaxWithLoss state_dict keys
        self.adaptive_log_softmax = nn.Module()
        self.adaptive_log_softmax.head = nn.Module()
        self.adaptive_log_softmax.head.weight = nn.Parameter(
            torch.empty(n_classes + len(cutoffs), in_features))
        self.adaptive_log_softmax.tail = nn.ModuleList()
        for i in range(len(cutoffs)):
            if i == 0:
                l = cutoffs[0]
            else:
                l = cutoffs[i] - cutoffs[i-1]
            h = int(in_features // (div_value ** (i+1)))
            seq = nn.Sequential()
            seq.add_module('0', nn.Linear(in_features, h, bias=False))
            seq.add_module('1', nn.Linear(h, l, bias=False))
            self.adaptive_log_softmax.tail.append(seq)

        # Initialize parameters if needed (matching reference initialization not required for state_dict keys)
        self._init_weights()

    def _init_weights(self):
        # Same initialization as nn.AdaptiveLogSoftmaxWithLoss
        nn.init.kaiming_uniform_(self.adaptive_log_softmax.head.weight, a=5 ** 0.5)
        for i in range(len(self.adaptive_log_softmax.tail)):
            nn.init.kaiming_uniform_(self.adaptive_log_softmax.tail[i][0].weight, a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.adaptive_log_softmax.tail[i][1].weight, a=5 ** 0.5)

    def forward(self, x, target=None, relu_input=None):
        if relu_input is None:
            relu_input = torch.randn(self.relu_input_shape, device=x.device, dtype=x.dtype)
        relu_output = cuda_extension.relu(relu_input)

        if target is not None:
            output, loss = cuda_extension.adaptive_log_softmax_loss(
                x, target,
                self.adaptive_log_softmax.head.weight,
                self.adaptive_log_softmax.tail[0][0].weight,
                self.adaptive_log_softmax.tail[0][1].weight,
                self.adaptive_log_softmax.tail[1][0].weight,
                self.adaptive_log_softmax.tail[1][1].weight,
                self.cutoffs[0],
                self.div_value,
                self.head_bias
            )
            return (output, loss), relu_output
        else:
            log_prob = cuda_extension.adaptive_log_softmax_log_prob(
                x,
                self.adaptive_log_softmax.head.weight,
                self.adaptive_log_softmax.tail[0][0].weight,
                self.adaptive_log_softmax.tail[0][1].weight,
                self.adaptive_log_softmax.tail[1][0].weight,
                self.adaptive_log_softmax.tail[1][1].weight,
                self.cutoffs[0],
                self.div_value,
                self.head_bias
            )
            return log_prob, relu_output

    def debug_forward(self, *inputs):
        # inputs: x, target, relu_input (as provided by get_inputs)
        x, target, relu_input = inputs
        if relu_input is None:
            relu_input = torch.randn(self.relu_input_shape, device=x.device, dtype=x.dtype)
        relu_output = cuda_extension.relu(relu_input)
        yield 'relu_output', relu_output

        if target is not None:
            output, loss = cuda_extension.adaptive_log_softmax_loss(
                x, target,
                self.adaptive_log_softmax.head.weight,
                self.adaptive_log_softmax.tail[0][0].weight,
                self.adaptive_log_softmax.tail[0][1].weight,
                self.adaptive_log_softmax.tail[1][0].weight,
                self.adaptive_log_softmax.tail[1][1].weight,
                self.cutoffs[0],
                self.div_value,
                self.head_bias
            )
            yield 'adaptive_loss_output', output
            yield 'adaptive_loss_loss', loss
            yield 'final', ((output, loss), relu_output)
        else:
            log_prob = cuda_extension.adaptive_log_softmax_log_prob(
                x,
                self.adaptive_log_softmax.head.weight,
                self.adaptive_log_softmax.tail[0][0].weight,
                self.adaptive_log_softmax.tail[0][1].weight,
                self.adaptive_log_softmax.tail[1][0].weight,
                self.adaptive_log_softmax.tail[1][1].weight,
                self.cutoffs[0],
                self.div_value,
                self.head_bias
            )
            yield 'adaptive_log_prob', log_prob
            yield 'final', (log_prob, relu_output)
