import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_classes, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, input, target, target_labels):
        # step_001: kl_loss = self.kl_div_loss(F.log_softmax(input, dim=1), F.softmax(target, dim=1))
        log_softmax_input = cuda_extension.log_softmax_forward(input, 1)
        softmax_target = cuda_extension.softmax_forward(target, 1)
        kl_loss = cuda_extension.kl_div_loss_forward(log_softmax_input, softmax_target)

        # step_002: ce_loss = self.cross_entropy_loss(input, target_labels)
        ce_loss = cuda_extension.cross_entropy_loss_forward(input, target_labels)

        # step_003: combined_loss = kl_loss + ce_loss
        combined_loss = cuda_extension.add_forward(kl_loss, ce_loss)

        # step_004: remainder_loss = torch.remainder(combined_loss, self.scaling_factor)
        remainder_loss = cuda_extension.remainder_forward(combined_loss, self.scaling_factor)

        return remainder_loss

    def debug_forward(self, input, target, target_labels):
        # step_001
        log_softmax_input = cuda_extension.log_softmax_forward(input, 1)
        softmax_target = cuda_extension.softmax_forward(target, 1)
        kl_loss = cuda_extension.kl_div_loss_forward(log_softmax_input, softmax_target)
        yield ("step_001", kl_loss)

        # step_002
        ce_loss = cuda_extension.cross_entropy_loss_forward(input, target_labels)
        yield ("step_002", ce_loss)

        # step_003
        combined_loss = cuda_extension.add_forward(kl_loss, ce_loss)
        yield ("step_003", combined_loss)

        # step_004
        remainder_loss = cuda_extension.remainder_forward(combined_loss, self.scaling_factor)
        yield ("step_004", remainder_loss)

        # step_005: final return
        yield ("step_005", remainder_loss)
