#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Unfold Kernel
// Each thread computes one output element (or uses grid-stride loop).
// Input:  x[N, C, H, W] contiguous
// Output: out[N, C*K_h*K_w, L] contiguous
// L = out_L = ((H - d_h*(K_h-1) - 1) / stride + 1) * ((W - d_w*(K_w-1) - 1) / stride + 1)
// stride = 1, padding = 0
// -----------------------------------------------------------------------------
__global__ void unfold_kernel(const float* __restrict__ x,
                              float* __restrict__ out,
                              int N, int C, int H, int W,
                              int K_h, int K_w,
                              int d_h, int d_w,
                              int out_H, int out_W,
                              int L) {
    int total = N * C * K_h * K_w * L;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int tmp = i;
        int l = tmp % L;
        tmp /= L;
        int kk = tmp % (K_h * K_w);
        tmp /= (K_h * K_w);
        int c = tmp % C;
        tmp /= C;
        int n = tmp;

        int kh = kk / K_w;
        int kw = kk % K_w;

        int col = l % out_W;
        int row = l / out_W;

        int in_h = row + kh * d_h;
        int in_w = col + kw * d_w;

        // Should be valid because L is computed from H,W and dilations.
        out[i] = x[((n * C + c) * H + in_h) * W + in_w];
    }
}

// -----------------------------------------------------------------------------
// BatchNorm statistics kernel (per-channel mean and variance)
// Input: input[N, C, L] contiguous
// Output: mean[C], var[C] (unbiased variance for update if update_running)
// Additionally update running_mean and running_var if non-null.
// mean = sum(x) / count
// var = sum((x-mean)^2) / count  (biased for normalization)
// unbiased_var = sum((x-mean)^2) / (count-1) for running update
// -----------------------------------------------------------------------------
__global__ void batch_norm_stats_kernel(const float* __restrict__ input,
                                        float* __restrict__ mean,
                                        float* __restrict__ var,
                                        float* __restrict__ running_mean,
                                        float* __restrict__ running_var,
                                        int N, int C, int L,
                                        float momentum,
                                        float eps) {
    int c = blockIdx.x;
    if (c >= C) return;

    int count = N * L;
    if (count == 0) {
        if (mean) mean[c] = 0.0f;
        if (var) var[c] = 0.0f;
        return;
    }

    extern __shared__ float sdata[]; // size = blockDim.x * 2
    float* s_sum = sdata;
    float* s_sum_sq = sdata + blockDim.x;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // grid-stride loop over all samples and positions
    int total = N * L;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        // input layout: (n * C + c) * L + l
        int n = idx / L;
        int l = idx - n * L;
        float val = input[(n * C + c) * L + l];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Reduction within block
    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = s_sum[0];
        float sum_sq = s_sum_sq[0];
        float batch_mean = sum / count;
        float batch_var = sum_sq / count - batch_mean * batch_mean;
        // Clamp to zero if slightly negative due to floating error
        if (batch_var < 0.0f) batch_var = 0.0f;

        if (mean) mean[c] = batch_mean;
        if (var) var[c] = batch_var;

        if (running_mean && running_var) {
            float unbiased_var = (count > 1) ? (sum_sq - (sum * sum) / count) / (count - 1) : 0.0f;
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * batch_mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
        }
    }
}

// -----------------------------------------------------------------------------
// BatchNorm normalization kernel (applies affine transformation)
// y = (x - mean) / sqrt(var + eps) * weight + bias
// input[N, C, L] contiguous, output same shape
// -----------------------------------------------------------------------------
__global__ void batch_norm_normalize_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            const float* __restrict__ mean,
                                            const float* __restrict__ var,
                                            const float* __restrict__ weight,
                                            const float* __restrict__ bias,
                                            int N, int C, int L,
                                            float eps) {
    int total = N * C * L;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int tmp = i;
        int l = tmp % L;
        tmp /= L;
        int c = tmp % C;
        int n = tmp / C;

        float mean_c = mean[c];
        float var_c = var[c];
        float inv_std = rsqrtf(var_c + eps);
        float gamma = (weight != nullptr) ? weight[c] : 1.0f;
        float beta = (bias != nullptr) ? bias[c] : 0.0f;

        float x = input[i];
        float y = (x - mean_c) * inv_std * gamma + beta;
        output[i] = y;
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C")
// -----------------------------------------------------------------------------

extern "C" void op_unfold_launcher(const float* x,
                                   float* out,
                                   int N, int C, int H, int W,
                                   int kernel_h, int kernel_w,
                                   int dilation_h, int dilation_w,
                                   cudaStream_t stream) {
    int out_H = (H - dilation_h * (kernel_h - 1) - 1) + 1;
    int out_W = (W - dilation_w * (kernel_w - 1) - 1) + 1;
    int L = out_H * out_W;
    int total = N * C * kernel_h * kernel_w * L;

    const int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535; // grid-stride loop handles rest

    unfold_kernel<<<grid, block, 0, stream>>>(x, out, N, C, H, W,
                                             kernel_h, kernel_w,
                                             dilation_h, dilation_w,
                                             out_H, out_W, L);
}

extern "C" void op_batch_norm_1d_training_launcher(const float* input,
                                                   float* output,
                                                   const float* weight,
                                                   const float* bias,
                                                   float* running_mean,
                                                   float* running_var,
                                                   int N, int C, int L,
                                                   float eps, float momentum,
                                                   cudaStream_t stream) {
    int count = N * C * L;
    if (count == 0) {
        return;
    }

    // Allocate temporary space for mean and var on device
    float *d_mean, *d_var;
    cudaMalloc(&d_mean, C * sizeof(float));
    cudaMalloc(&d_var, C * sizeof(float));
    if (d_mean == nullptr || d_var == nullptr) {
        // Allocation failed; return without computing, but this should not happen.
        // The binding layer should call cudaGetLastError to see the error.
        return;
    }

    // Compute statistics, one block per channel, with shared memory reduction.
    const int block = 256;
    // Use up to 1024 threads for reduction, but block size 256 is enough and safe.
    int smem_size = 2 * block * sizeof(float);
    batch_norm_stats_kernel<<<C, block, smem_size, stream>>>(
        input, d_mean, d_var, running_mean, running_var,
        N, C, L, momentum, eps);

    // Normalize using computed mean and var
    int total = N * C * L;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    batch_norm_normalize_kernel<<<grid, block, 0, stream>>>(
        input, output, d_mean, d_var, weight, bias,
        N, C, L, eps);

    // Free temporary memory after kernel launch (asynchronously safe)
    cudaFree(d_mean);
    cudaFree(d_var);
}
