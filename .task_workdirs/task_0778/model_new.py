import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, search_values, max_pool_kernel_size):
        super().__init__()
        self.isin_values = nn.Parameter(torch.tensor(search_values), requires_grad=False)
        self.max_pool_kernel_size = max_pool_kernel_size

    def forward(self, x):
        # Step 1: mask = torch.isin(x, self.isin_values)
        # isin_values is a CPU parameter; custom op expects CUDA tensor,
        # so we move it to CUDA. Since x is already CUDA, this transfer is
        # necessary and is allowed as tensor creation/movement, not a torch op.
        values_cuda = self.isin_values.cuda()
        mask = cuda_extension.isin_custom(x, values_cuda)

        # Step 2: x = self.max_pool(x)
        # MaxPool1d with kernel_size=self.max_pool_kernel_size, stride=same,
        # padding=0, dilation=1, ceil_mode=False (as in original nn.MaxPool1d(kernel_size))
        x = cuda_extension.maxpool1d_custom(
            x,
            self.max_pool_kernel_size,
            self.max_pool_kernel_size,
            0,
            1,
            False
        )

        return x, mask

    def debug_forward(self, x):
        values_cuda = self.isin_values.cuda()
        mask = cuda_extension.isin_custom(x, values_cuda)
        yield 'step_001', mask

        pooled = cuda_extension.maxpool1d_custom(
            x,
            self.max_pool_kernel_size,
            self.max_pool_kernel_size,
            0,
            1,
            False
        )
        yield 'step_002', pooled

        yield 'step_003', (pooled, mask)
