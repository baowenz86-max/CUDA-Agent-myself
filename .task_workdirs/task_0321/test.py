import torch

import cuda_extension
import decompiled_cuda_extension


x = torch.randn(
    1024,
    device="cuda"
)


out1 = cuda_extension.ceil_transpose_group_norm(x)

out2 = decompiled_cuda_extension.ceil_transpose_group_norm(x)


torch.testing.assert_close(
    out1,
    out2,
    atol=1e-5,
    rtol=1e-5
)


print("PASS")