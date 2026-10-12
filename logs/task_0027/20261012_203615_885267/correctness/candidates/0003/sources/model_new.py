import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self,
        num_features,
        batch_size,
        convTranspose_in_channels,
        convTranspose_out_channels,
        convTranspose_kernel_size,
        convTranspose_stride,
        convTranspose_padding,
        convTranspose_output_padding,
    ):
        super().__init__()
        # Preserve original state_dict keys: sync_batchnorm.* and convTranspose.*
        self.sync_batchnorm = nn.Module()
        self.sync_batchnorm.weight = nn.Parameter(torch.ones(num_features))
        self.sync_batchnorm.bias = nn.Parameter(torch.zeros(num_features))
        self.sync_batchnorm.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.sync_batchnorm.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)
        self.sync_batchnorm.num_batches_tracked = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)

        self.convTranspose = nn.Module()
        self.convTranspose.weight = nn.Parameter(
            torch.empty(convTranspose_in_channels, convTranspose_out_channels, *(
                (convTranspose_kernel_size, convTranspose_kernel_size)
                if isinstance(convTranspose_kernel_size, int)
                else tuple(convTranspose_kernel_size)
            ))
        )
        self.convTranspose.bias = nn.Parameter(torch.empty(convTranspose_out_channels))

        self.batch_size = batch_size
        self.num_features = num_features
        self.convTranspose_in_channels = convTranspose_in_channels
        self.convTranspose_out_channels = convTranspose_out_channels
        # kernel_size may be int or tuple; normalize to tuple
        if isinstance(convTranspose_kernel_size, int):
            self.kernel_h = self.kernel_w = convTranspose_kernel_size
        else:
            self.kernel_h, self.kernel_w = convTranspose_kernel_size
        # stride, padding, output_padding may be int or tuple
        if isinstance(convTranspose_stride, int):
            self.stride_h = self.stride_w = convTranspose_stride
        else:
            self.stride_h, self.stride_w = convTranspose_stride
        if isinstance(convTranspose_padding, int):
            self.padding_h = self.padding_w = convTranspose_padding
        else:
            self.padding_h, self.padding_w = convTranspose_padding
        if isinstance(convTranspose_output_padding, int):
            self.output_padding_h = self.output_padding_w = convTranspose_output_padding
        else:
            self.output_padding_h, self.output_padding_w = convTranspose_output_padding

    def forward(self, x, mat1, mat2):
        # 1. SyncBatchNorm in eval mode: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
        x_bn = cuda_extension.batchnorm_eval_forward(
            x,
            self.sync_batchnorm.weight,
            self.sync_batchnorm.bias,
            self.sync_batchnorm.running_mean,
            self.sync_batchnorm.running_var,
            1e-5,
        )
        # 2. Batch matrix multiply
        bmm_result = cuda_extension.bmm_forward(mat1, mat2)
        # 3. ConvTranspose2d on reshaped bmm_result: view(batch, in_channels, 1, -1)
        # seq_len = bmm_result.shape[-1] (same as mat2.shape[-1])
        seq_len = bmm_result.shape[-1]
        out = cuda_extension.conv_transpose2d_forward(
            bmm_result,
            self.convTranspose.weight,
            self.convTranspose.bias,
            self.batch_size,
            self.convTranspose_in_channels,
            self.convTranspose_out_channels,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.output_padding_h,
            self.output_padding_w,
            seq_len,
        )
        return out

    def debug_forward(self, x, mat1, mat2):
        # Yield checkpoints in exact order and ids as specified by reference trace.
        x_bn = cuda_extension.batchnorm_eval_forward(
            x,
            self.sync_batchnorm.weight,
            self.sync_batchnorm.bias,
            self.sync_batchnorm.running_mean,
            self.sync_batchnorm.running_var,
            1e-5,
        )
        yield "step_001", x_bn

        bmm_result = cuda_extension.bmm_forward(mat1, mat2)
        yield "step_002", bmm_result

        seq_len = bmm_result.shape[-1]
        out = cuda_extension.conv_transpose2d_forward(
            bmm_result,
            self.convTranspose.weight,
            self.convTranspose.bias,
            self.batch_size,
            self.convTranspose_in_channels,
            self.convTranspose_out_channels,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.output_padding_h,
            self.output_padding_w,
            seq_len,
        )
        yield "step_003", out
        yield "step_004", out
