import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, min_val):
        super().__init__()
        self.num_features = num_features
        self.min_val = min_val

        # 保持 state_dict 键与参考模型一致：batch_norm.weight/bias/running_mean/running_var/num_batches_tracked
        self.batch_norm = nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        # 参考语句 1: x = self.batch_norm(x)  （推理模式，training=False）
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            1e-05
        )
        # 参考语句 2: x = torch.clamp(x, min=self.min_val)
        x = cuda_extension.clamp_min_forward(x, self.min_val)
        # 参考语句 3: return x
        return x

    def debug_forward(self, x):
        # 按 task_contract checkpoints 顺序 yield (id, value)
        norm_out = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            1e-05
        )
        yield 'step_001', norm_out

        clamped_out = cuda_extension.clamp_min_forward(norm_out, self.min_val)
        yield 'step_002', clamped_out

        yield 'step_003', clamped_out
