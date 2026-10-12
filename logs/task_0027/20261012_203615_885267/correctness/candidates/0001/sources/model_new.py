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
        # Use nn.SyncBatchNorm to preserve state_dict keys and running stats behavior.
        # We do not call its forward; instead we use our custom op with its parameters.
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
        # Store configuration for custom conv op
        self.convTranspose_in_channels = convTranspose_in_channels
        self.convTranspose_out_channels = convTranspose_out_channels
        self.convTranspose_kernel_size = convTranspose_kernel_size
        self.convTranspose_stride = convTranspose_stride
        self.convTranspose_padding = convTranspose_padding
        self.convTranspose_output_padding = convTranspose_output_padding

    def forward(self, x, mat1, mat2):
        # Step 1: SyncBatchNorm eval mode
        x = cuda_extension.batchnorm_eval_forward(
            x,
            self.sync_batchnorm.weight,
            self.sync_batchnorm.bias,
            self.sync_batchnorm.running_mean,
            self.sync_batchnorm.running_var,
            self.sync_batchnorm.eps,
        )

        # Step 2: batch matmul
        bmm_result = cuda_extension.bmm_forward(mat1, mat2)

        # Step 3: view and convTranspose
        # bmm_result shape: [batch_size, seq_len, convTranspose_in_channels]
        # view to [batch_size, convTranspose_in_channels, 1, seq_len]
        # Note: seq_len = bmm_result.shape[1]
        x = cuda_extension.conv_transpose2d_forward(
            bmm_result,
            self.convTranspose.weight,
            self.convTranspose.bias,
            self.batch_size,
            self.convTranspose_in_channels,
            self.convTranspose_out_channels,
            self.convTranspose_kernel_size,
            self.convTranspose_stride,
            self.convTranspose_padding,
            self.convTranspose_output_padding,
        )
        return x

    def debug_forward(self, x, mat1, mat2):
        # Step 1
        x = cuda_extension.batchnorm_eval_forward(
            x,
            self.sync_batchnorm.weight,
            self.sync_batchnorm.bias,
            self.sync_batchnorm.running_mean,
            self.sync_batchnorm.running_var,
            self.sync_batchnorm.eps,
        )
        yield 'step_001', x

        # Step 2
        bmm_result = cuda_extension.bmm_forward(mat1, mat2)
        yield 'step_002', bmm_result

        # Step 3
        x = cuda_extension.conv_transpose2d_forward(
            bmm_result,
            self.convTranspose.weight,
            self.convTranspose.bias,
            self.batch_size,
            self.convTranspose_in_channels,
            self.convTranspose_out_channels,
            self.convTranspose_kernel_size,
            self.convTranspose_stride,
            self.convTranspose_padding,
            self.convTranspose_output_padding,
        )
        yield 'step_003', x

        # Step 4: final return
        yield 'step_004', x
