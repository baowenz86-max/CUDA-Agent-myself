#include <cuda_runtime.h>
#include <cstdint>

// Custom max over dimension 1 with keepdim=True
// Input: x shape [N, C, H, W], stored in any strides (we only accept contiguous for now, but use strides for generality)
// Output: values [N, 1, H, W] float32, indices [N, 1, H, W] int64
// Reduction over C with tie-breaking to first occurrence.
__global__ void max_dim1_kernel(const float* __restrict__ x,
                                float* __restrict__ values,
                                int64_t* __restrict__ indices,
                                int N, int C, int H, int W,
                                int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    int total = N * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int n = idx / (H * W);
    int rem = idx % (H * W);
    int h = rem / W;
    int w = rem % W;

    const float* x_n = x + n * stride_n;
    float max_val = x_n[0 * stride_c + h * stride_h + w * stride_w];
    int64_t max_idx = 0;
    for (int c = 1; c < C; ++c) {
        float val = x_n[c * stride_c + h * stride_h + w * stride_w];
        if (val > max_val) {
            max_val = val;
            max_idx = c;
        }
    }
    int out_idx = n * (H * W) + h * W + w;
    values[out_idx] = max_val;
    indices[out_idx] = max_idx;
}

// Custom layer normalization over last dimension (C)
// Input x shape [N, 1, H, W] interpreted as [N, H, W, C]
// weight and bias shape [C]. Output same shape.
// We assume contiguous layout for x: N*H*W*C elements, where C = W dimension? Actually we know from contract:
// Input to layer_norm is max output shape [512,1,1024,64], so N=512, H=1024, W=64, and normalized_shape is [64] = W.
// We treat last dim size as C_norm = W = 64. For general, we accept C_norm as W.
__global__ void layer_norm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ weight,
                                  const float* __restrict__ bias,
                                  float* __restrict__ y,
                                  int N, int H, int W, int C_norm,
                                  double eps) {
    int total = N * H * W; // number of normalization groups
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const float* x_group = x + idx * C_norm; // contiguous assumption
    float sum = 0.0f;
    for (int i = 0; i < C_norm; ++i) {
        sum += x_group[i];
    }
    float mean = sum / C_norm;
    float sq_sum = 0.0f;
    for (int i = 0; i < C_norm; ++i) {
        float diff = x_group[i] - mean;
        sq_sum += diff * diff;
    }
    float var = sq_sum / C_norm;
    float inv_std = rsqrtf(var + (float)eps);

    float* y_group = y + idx * C_norm;
    for (int i = 0; i < C_norm; ++i) {
        y_group[i] = (x_group[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// Launcher for custom_max_dim1_keepdim
extern "C" void custom_max_dim1_keepdim_launcher(const float* x,
                                                float* values,
                                                int64_t* indices,
                                                int N, int C, int H, int W,
                                                int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                                cudaStream_t stream) {
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    max_dim1_kernel<<<blocks, threads, 0, stream>>>(x, values, indices, N, C, H, W,
                                                    stride_n, stride_c, stride_h, stride_w);
}

// Launcher for custom_layer_norm
extern "C" void custom_layer_norm_launcher(const float* x,
                                          const float* weight,
                                          const float* bias,
                                          float* y,
                                          int N, int H, int W, int C_norm,
                                          double eps,
                                          cudaStream_t stream) {
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    layer_norm_kernel<<<blocks, threads, 0, stream>>>(x, weight, bias, y, N, H, W, C_norm, eps);
}
