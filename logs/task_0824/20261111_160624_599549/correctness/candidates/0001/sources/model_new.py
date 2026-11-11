import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, kernel_size, stride, padding, dilation):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.max_pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        # Step 1: BatchNorm2d
        x = cuda_extension.batch_norm_2d_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
            self.training
        )
        # Step 2: permute(0,2,1,3).contiguous()
        x = cuda_extension.permute_contiguous_4d(x, 0, 2, 1, 3)
        # Step 3: view(x.size(0), x.size(1), -1)
        x = cuda_extension.view_4d(x, x.size(0), x.size(1), -1)
        # Step 4: max_pool1d
        x = cuda_extension.max_pool1d_forward(
            x,
            self.max_pool.kernel_size,
            self.max_pool.stride,
            self.max_pool.padding,
            self.max_pool.dilation,
            self.max_pool.ceil_mode
        )
        # Step 5: view(x.size(0), x.size(1), -1, 1)
        x = cuda_extension.view_5d(x, x.size(0), x.size(1), -1, 1)
        # Step 6: permute(0,2,1,3).contiguous()
        x = cuda_extension.permute_contiguous_4d_v2(x, 0, 2, 1, 3)
        return x

    def debug_forward(self, x):
        # Step 1: BatchNorm2d
        out = cuda_extension.batch_norm_2d_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
            self.training
        )
        yield ('step_001', out)
        # Step 2: permute(0,2,1,3).contiguous()
        out = cuda_extension.permute_contiguous_4d(out, 0, 2, 1, 3)
        yield ('step_002', out)
        # Step 3: view(x.size(0), x.size(1), -1)
        out = cuda_extension.view_4d(out, out.size(0), out.size(1), -1)
        yield ('step_003', out)
        # Step 4: max_pool1d
        out = cuda_extension.max_pool1d_forward(
            out,
            self.max_pool.kernel_size,
            self.max_pool.stride,
            self.max_pool.padding,
            self.max_pool.dilation,
            self.max_pool.ceil_mode
        )
        yield ('step_004', out)
        # Step 5: view(x.size(0), x.size(1), -1, 1)
        out = cuda_extension.view_5d(out, out.size(0), out.size(1), -1, 1)
        yield ('step_005', out)
        # Step 6: permute(0,2,1,3).contiguous()
        out = cuda_extension.permute_contiguous_4d_v2(out, 0, 2, 1, 3)
        yield ('step_006', out)
        # Step 7: return
        yield ('step_007', out)
