import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # 不创建任何子模块或参数，保持 state_dict 为空

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.instance_norm_forward(x, self.num_features, self.eps)

    def debug_forward(self, x: torch.Tensor):
        y = cuda_extension.instance_norm_forward(x, self.num_features, self.eps)
        yield "step_001", y
