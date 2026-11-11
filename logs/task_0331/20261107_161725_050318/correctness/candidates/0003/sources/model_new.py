import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, bias_shape):
        super().__init__()
        self.fc = nn.Linear(in_features, in_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Step 1: linear
        x_linear = cuda_extension.linear_forward(x, self.fc.weight, self.fc.bias)
        # Step 2: gelu
        x_gelu = cuda_extension.gelu_forward(x_linear)
        # Step 3: comparison with threshold 0.5
        x_gt = cuda_extension.gt_forward(x_gelu, 0.5)
        # Step 4: broadcasting addition
        x_out = cuda_extension.add_forward(x_gt, self.bias)
        return x_out

    def debug_forward(self, x):
        # Must yield intermediate values according to checkpoints
        x_linear = cuda_extension.linear_forward(x, self.fc.weight, self.fc.bias)
        yield ('step_001', x_linear)
        x_gelu = cuda_extension.gelu_forward(x_linear)
        yield ('step_002', x_gelu)
        # threshold creation: In reference, threshold = torch.tensor(0.5) is a CPU scalar tensor.
        # But we cannot create torch tensors in model_new.py. However, we need to yield it.
        # We can create a tensor using torch.tensor? The instruction says: '禁止使用 torch 算子实现本应下沉到自定义扩展的计算。' but creating a constant scalar tensor may be considered tensor creation, not an operation.
        # To be safe, we can create a tensor using torch.tensor(0.5) in debug_forward only for yielding, but that would violate? Actually model_new.py can use torch.tensor creation (allowed per SKILL: 'torch.tensor creation').
        # Forward: we can create a CPU tensor with torch.tensor(0.5) and pass to gt op? But gt op will be custom and might accept scalar float parameter. In forward, we choose to pass scalar float directly to avoid tensor creation.
        # In debug_forward, we need to yield the actual threshold tensor as in reference. So we can do:
        threshold = torch.tensor(0.5)
        yield ('step_003', threshold)
        x_gt = cuda_extension.gt_forward(x_gelu, 0.5)
        yield ('step_004', x_gt)
        x_out = cuda_extension.add_forward(x_gt, self.bias)
        yield ('step_005', x_out)
        yield ('step_006', x_out)
        return x_out