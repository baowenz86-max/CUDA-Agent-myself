import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scale_factor):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        # Step 1: x = torch.addmm(self.bias, x, self.weight.t())
        # = bias + x @ weight.t()
        # custom op: addmm_custom(bias, x, weight) -> [batch_size, hidden_size]
        x = cuda_extension.addmm_custom(self.bias, x, self.weight)

        # Step 2: x = torch.unsqueeze(x, 0)  # Adding a batch dimension
        # custom op: unsqueeze_custom(x, dim=0) -> [1, batch_size, hidden_size]
        x = cuda_extension.unsqueeze_custom(x, 0)

        # Step 3: x = F.interpolate(x, scale_factor=self.scale_factor, mode="linear", align_corners=False)
        # custom op: interpolate_linear_custom(x, scale_factor, align_corners=False) -> [1, batch_size, output_size]
        x = cuda_extension.interpolate_linear_custom(x, self.scale_factor, False)
        return x

    def debug_forward(self, x):
        # Same calculations as forward, but yield intermediate checkpoints.
        # Step 1
        x1 = cuda_extension.addmm_custom(self.bias, x, self.weight)
        yield ("step_001", x1)

        # Step 2
        x2 = cuda_extension.unsqueeze_custom(x1, 0)
        yield ("step_002", x2)

        # Step 3
        x3 = cuda_extension.interpolate_linear_custom(x2, self.scale_factor, False)
        yield ("step_003", x3)

        # Final return
        yield ("step_004", x3)
