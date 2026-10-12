import torch
import torch.nn as nn


class Model(nn.Module):
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

    def forward(self, x, mat1, mat2):
        x = self.sync_batchnorm(x)
        bmm_result = torch.bmm(mat1, mat2)
        x = self.convTranspose(bmm_result.view(self.batch_size, convTranspose_in_channels, 1, -1))
        return x


# Hyperparameters
num_features = 128
batch_size = 512
seq_len = 256  # Increased from 32 to increase BMM and ConvTranspose computation
convTranspose_in_channels = 128
convTranspose_out_channels = 64
convTranspose_kernel_size = 2
convTranspose_stride = 1
convTranspose_padding = 0
convTranspose_output_padding = 0


def get_inputs():
    # Increased spatial dimensions from 32x32 to 64x64 for SyncBatchNorm
    x = torch.randn(batch_size, num_features, 64, 64)
    mat1 = torch.randn(batch_size, seq_len, convTranspose_in_channels)
    mat2 = torch.randn(batch_size, convTranspose_in_channels, convTranspose_in_channels)
    return [x, mat1, mat2]


def get_init_inputs():
    return [
        num_features,
        batch_size,
        convTranspose_in_channels,
        convTranspose_out_channels,
        convTranspose_kernel_size,
        convTranspose_stride,
        convTranspose_padding,
        convTranspose_output_padding,
    ]