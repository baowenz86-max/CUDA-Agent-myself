#include <cuda_runtime.h>
#include <cstdint>

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
    // Output tensors are contiguous with shape [N,1,H,W]; stride is [H*W, H*W, W, 1]
    // So flat output index is n*(H*W) + h*W + w
    int out_idx = n * (H * W) + h * W + w;
    values[out_idx] = max_val;
    indices[out_idx] = max_idx;
}

__global__ void layer_norm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ weight,
                                  const float* __restrict__ bias,
                                  float* __restrict__ y,
                                  int N, int H, int W, int C_norm,
                                  double eps) {
    // Correct number of normalization groups: N * 1 * H = N * H
    int total = N * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // x is contiguous [N,1,H,W], so group idx corresponds to n=idx/H, h=idx%H,
    // offset = (n * (1*H*W) + 0*(H*W) + h*W) = (n * H + h) * W = idx * W
    const float* x_group = x + idx * W;
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

    float* y_group = y + idx * W;
    for (int i = 0; i < C_norm; ++i) {
        y_group[i] = (x_group[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

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

extern "C" void custom_layer_norm_launcher(const float* x,
                                          const float* weight,
                                          const float* bias,
                                          float* y,
                                          int N, int H, int W, int C_norm,
                                          double eps,
                                          cudaStream_t stream) {
    int total = N * H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    layer_norm_kernel<<<blocks, threads, 0, stream>>>(x, weight, bias, y, N, H, W, C_norm, eps);
}
