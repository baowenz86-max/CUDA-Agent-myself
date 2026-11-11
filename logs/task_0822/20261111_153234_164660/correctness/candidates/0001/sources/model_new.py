import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value, head_bias, relu_input_shape):
        super().__init__()
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features, n_classes, cutoffs, div_value, head_bias
        )
        self.relu_input_shape = relu_input_shape

    def forward(self, x, target=None, relu_input=None):
        if relu_input is None:
            relu_input = cuda_extension.randn_forward(self.relu_input_shape)
        relu_output = cuda_extension.relu_forward(relu_input)

        if target is not None:
            adaptive_result = cuda_extension.adaptive_log_softmax_with_loss_forward(
                x,
                target,
                self.adaptive_log_softmax.head.weight,
                self.adaptive_log_softmax.tail[0][0].weight,
                self.adaptive_log_softmax.tail[0][1].weight,
                self.adaptive_log_softmax.tail[1][0].weight,
                self.adaptive_log_softmax.tail[1][1].weight,
                self.adaptive_log_softmax.cutoffs[0],
                self.adaptive_log_softmax.cutoffs[1],
                self.adaptive_log_softmax.div_value,
            )
            return adaptive_result, relu_output
        else:
            log_prob_result = cuda_extension.adaptive_log_softmax_log_prob_forward(
                x,
                self.adaptive_log_softmax.head.weight,
                self.adaptive_log_softmax.tail[0][0].weight,
                self.adaptive_log_softmax.tail[0][1].weight,
                self.adaptive_log_softmax.tail[1][0].weight,
                self.adaptive_log_softmax.tail[1][1].weight,
                self.adaptive_log_softmax.cutoffs[0],
                self.adaptive_log_softmax.cutoffs[1],
                self.adaptive_log_softmax.div_value,
            )
            return log_prob_result, relu_output

    def debug_forward(self, *inputs):
        x, target, relu_input = inputs

        yielded_relu_input = (
            cuda_extension.randn_forward(self.relu_input_shape)
            if relu_input is None
            else relu_input
        )
        yield ("checkpoint_000", yielded_relu_input)

        relu_output = cuda_extension.relu_forward(yielded_relu_input)
        yield ("checkpoint_001", relu_output)

        if target is not None:
            asm_output = cuda_extension.adaptive_log_softmax_with_loss_forward(
                x,
                target,
                self.adaptive_log_softmax.head.weight,
                self.adaptive_log_softmax.tail[0][0].weight,
                self.adaptive_log_softmax.tail[0][1].weight,
                self.adaptive_log_softmax.tail[1][0].weight,
                self.adaptive_log_softmax.tail[1][1].weight,
                self.adaptive_log_softmax.cutoffs[0],
                self.adaptive_log_softmax.cutoffs[1],
                self.adaptive_log_softmax.div_value,
            )
            yield ("checkpoint_002", asm_output)
            result = (asm_output, relu_output)
        else:
            log_prob_output = cuda_extension.adaptive_log_softmax_log_prob_forward(
                x,
                self.adaptive_log_softmax.head.weight,
                self.adaptive_log_softmax.tail[0][0].weight,
                self.adaptive_log_softmax.tail[0][1].weight,
                self.adaptive_log_softmax.tail[1][0].weight,
                self.adaptive_log_softmax.tail[1][1].weight,
                self.adaptive_log_softmax.cutoffs[0],
                self.adaptive_log_softmax.cutoffs[1],
                self.adaptive_log_softmax.div_value,
            )
            yield ("checkpoint_002", log_prob_output)
            result = (log_prob_output, relu_output)

        yield ("checkpoint_003", result)
