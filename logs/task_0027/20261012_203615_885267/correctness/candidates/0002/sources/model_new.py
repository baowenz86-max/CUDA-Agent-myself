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
        # Preserve original submodule structure and state_dict keys
        self.sync_batchnorm = nn.SyncBatchNorm(num_features)
        self.convTranspose = nn.ConvTranspose2d(
            convTranspose_in_channels,
            convTranspose_out_channels,
            convTranspose_kernel_size,
            stride=convTranspose_stride,
            padding=convTranspose_padding,
            output_padding=convTranspose_output_padding,
        )
        self.batch_size = batch_size
        self.num_features = num_features
        self.convTranspose_in_channels = convTranspose_in_channels
        self.convTranspose_out_channels = convTranspose_out_channels
        self.convTranspose_kernel_size = convTranspose_kernel_size
        self.convTranspose_stride = convTranspose_stride
        self.convTranspose_padding = convTranspose_padding
        self.convTranspose_output_padding = convTranspose_output_padding

    def forward(self, x, mat1, mat2):
        # Step 1: SyncBatchNorm eval using running stats
        bn_weight = self.sync_batchnorm.weight
        bn_bias = self.sync_batchnorm.bias
        running_mean = self.sync_batchnorm.running_mean
        running_var = self.sync_batchnorm.running_var
        eps = self.sync_batchnorm.eps
        x_bn = cuda_extension.batchnorm_eval_forward(
            x, bn_weight, bn_bias, running_mean, running_var, eps
        )

        # Step 2: batch matrix multiply
        bmm_result = cuda_extension.bmm_forward(mat1, mat2)

        # Step 3: reshape bmm_result to conv input shape
        # bmm_result shape: [batch_size, seq_len, in_channels]
        # Need [batch_size, in_channels, 1, seq_len] for conv with height 1 width seq_len
        conv_input = bmm_result.view(
            self.batch_size, self.convTranspose_in_channels, 1, -1
        )

        # Step 4: ConvTranspose2d
        # Parameters passed:
        #   conv_input, weight, bias,
        #   batch_size, in_channels, out_channels,
        #   kernel_h, kernel_w,
        #   stride_h, stride_w,
        #   padding_h, padding_w,
        #   output_padding_h, output_padding_w,
        #   seq_len
        x_conv = cuda_extension.conv_transpose2d_forward(
            conv_input,
            self.convTranspose.weight,
            self.convTranspose.bias,
            self.batch_size,
            self.convTranspose_in_channels,
            self.convTranspose_out_channels,
            self.convTranspose_kernel_size[0],
            self.convTranspose_kernel_size[1],
            self.convTranspose_stride[0],
            self.convTranspose_stride[1],
            self.convTranspose_padding[0],
            self.convTranspose_padding[1],
            self.convTranspose_output_padding[0],
            self.convTranspose_output_padding[1],
            conv_input.size(-1),  # seq_len
        )
        return x_conv

    def debug_forward(self, x, mat1, mat2):
        # Must yield (checkpoint_id, value) in order: step_001, step_002, step_003, step_004
        bn_weight = self.sync_batchnorm.weight
        bn_bias = self.sync_batchnorm.bias
        running_mean = self.sync_batchnorm.running_mean
        running_var = self.sync_batchnorm.running_var
        eps = self.sync_batchnorm.eps

        x_bn = cuda_extension.batchnorm_eval_forward(
            x, bn_weight, bn_bias, running_mean, running_var, eps
        )
        yield "step_001", x_bn

        bmm_result = cuda_extension.bmm_forward(mat1, mat2)
        yield "step_002", bmm_result

        conv_input = bmm_result.view(
            self.batch_size, self.convTranspose_in_channels, 1, -1
        )
        x_conv = cuda_extension.conv_transpose2d_forward(
            conv_input,
            self.convTranspose.weight,
            self.convTranspose.bias,
            self.batch_size,
            self.convTranspose_in_channels,
            self.convTranspose_out_channels,
            self.convTranspose_kernel_size[0],
            self.convTranspose_kernel_size[1],
            self.convTranspose_stride[0],
            self.convTranspose_stride[1],
            self.convTranspose_padding[0],
            self.convTranspose_padding[1],
            self.convTranspose_output_padding[0],
            self.convTranspose_output_padding[1],
            conv_input.size(-1),
        )
        yield "step_003", x_conv
        yield "step_004", x_conv
