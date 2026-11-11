#include <cuda_runtime.h>

// BatchNorm2d evaluation-mode elementwise normalization kernel.
// Computes y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// for each element. Handles arbitrary strides for x. weight/bias/running_mean/running_var
// are expected contiguous and non-null if affine=True; but kernel will also work if pointers
// are nullptr (treated as weight=1, bias=0). running_mean/var are required, must not be null.
__global__ void norm_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int64_t numel,
    int64_t C,
    int64_t spatial_size,  // H * W
    const int64_t* __restrict__ x_strides  // array of 4: N_stride, C_stride, H_stride, W_stride
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; i < numel; i += stride) {
        // Compute 4D indices from linear index using x strides.
        // Since x may be non-contiguous but shape is [N,C,H,W].
        // We need channel index c = (i / (H*W)) % C, but careful with non-contiguous x.
        // General method: compute n,c,h,w from linear index assuming contiguous logical layout,
        // then compute memory offset via x_strides. This works because the output is also
        // shaped [N,C,H,W] and we want same logical position.
        int64_t rem = i;
        int64_t w = rem % spatial_size;
        rem /= spatial_size;
        int64_t c = rem % C;
        rem /= C;
        int64_t n = rem;  // N dimension, no need for h separately since spatial_size = H*W
        // Actually need h and w for strides, but spatial_size includes both. We can reconstruct
        // h and w from w index? For strides, we need separate h,w. We can compute h = w / W, w_idx = w % W if W known.
        // But we only have spatial_size = H*W. We need W to split h,w. Pass W separately? 
        // Let's modify: pass W as well, and H as spatial_size/W. Use explicit H,W.
    }
}
