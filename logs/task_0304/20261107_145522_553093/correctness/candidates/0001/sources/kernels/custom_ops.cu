#include <cuda_runtime.h>
#include <cstdint>

// Custom CUDA kernels for view_as_real and SyncBatchNorm (eval mode)
//
// view_as_real_forward:
//   Input: complex64 tensor shape [N,C,H] (contiguous)
//   Output: float32 tensor shape [N,C,H,2] (contiguous)
//   Each complex number c = a + bi is copied to output[...,0] = a, output[...,1] = b.
//
// sync_batchnorm_forward:
//   Input: float32 tensor x shape [N,C,H,W]
//   weight, bias: float32 tensor shape [C]
//   running_mean, running_var: float32 tensor shape [C]
//   eps: float scalar
//   Output: float32 tensor same shape as x, normalized using running stats
//   (eval mode): y = ((x - running_mean[c]) / sqrt(running_var[c] + eps)) * weight[c] + bias[c]
//   According to reference semantics, we use batch statistics? 
//   The task contract says 'using batch statistics' but reference model in training=false uses running stats.
//   We follow the provided reference_steps and model forward: SyncBatchNorm in eval mode.
//   The launcher receives running_mean/running_var from binding layer (which can access model buffers).
//
// All launchers are extern "C" and use raw pointers and cudaStream_t only.

namespace {

__global__ void view_as_real_kernel(const float2* __restrict__ input,
                                    float* __restrict__ output,
                                    int64_t total_complex) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_complex; i += stride) {
        float2 val = input[i];
        output[2 * i] = val.x;
        output[2 * i + 1] = val.y;
    }
}

// Reduction kernel to compute per-channel sum and sum of squares
// Grid: (num_chunks_per_channel, C), each block handles a contiguous chunk of HW plane for a given channel.
// Chunk size fixed 256, grid x dimension = ceil(HW / 256).
// Each block writes partial sums to partial_sum and partial_sqsum arrays.
__global__ void compute_channel_stats_kernel(const float* __restrict__ x,
                                             int64_t N,
                                             int64_t C,
                                             int64_t H,
                                             int64_t W,
                                             float* __restrict__ partial_sum,
                                             float* __restrict__ partial_sqsum) {
    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sqsum = smem + blockDim.x;

    int c = blockIdx.y;
    int chunk = blockIdx.x;
    int64_t HW = H * W;
    int chunk_size = blockDim.x;
    int64_t start = (int64_t)chunk * chunk_size;
    int64_t end = min((int64_t)(chunk + 1) * chunk_size, HW);

    float local_sum = 0.0f;
    float local_sqsum = 0.0f;

    for (int64_t n = 0; n < N; ++n) {
        const float* base = x + n * C * H * W + c * H * W;
        for (int64_t hw = start + threadIdx.x; hw < end; hw += chunk_size) {
            float v = base[hw];
            local_sum += v;
            local_sqsum += v * v;
        }
    }

    s_sum[threadIdx.x] = local_sum;
    s_sqsum[threadIdx.x] = local_sqsum;
    __syncthreads();

    // block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int index = c * gridDim.x + chunk;
        partial_sum[index] = s_sum[0];
        partial_sqsum[index] = s_sqsum[0];
    }
}

// Finalize per-channel mean and variance from partial sums
__global__ void finalize_stats_kernel(const float* __restrict__ partial_sum,
                                      const float* __restrict__ partial_sqsum,
                                      int64_t C,
                                      int64_t chunks_per_channel,
                                      float* __restrict__ mean,
                                      float* __restrict__ var) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    float sqsum = 0.0f;
    const float* ps = partial_sum + c * chunks_per_channel;
    const float* pq = partial_sqsum + c * chunks_per_channel;
    for (int i = 0; i < chunks_per_channel; ++i) {
        sum += ps[i];
        sqsum += pq[i];
    }
    float mu = sum / (float)(N_total_count); // replaced by actual count later
    // To avoid division by zero, but count is large
    mu = sum / (float)total_count;
    float var = sqsum / (float)total_count - mu * mu;
    mean[c] = mu;
    var[c] = (var > 0.0f) ? var : 0.0f;
}

// Normalization kernel using precomputed mean/var, weight/bias, eps
__global__ void apply_batchnorm_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int64_t N,
                                       int64_t C,
                                       int64_t H,
                                       int64_t W,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ var,
                                       const float* __restrict__ weight,
                                       const float* __restrict__ bias,
                                       float eps) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H * W;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        // derive c
        int64_t plane = H * W;
        int64_t n = i / (C * plane);
        int64_t rem = i % (C * plane);
        int c = rem / plane;
        float v = x[i];
        float normalized = (v - mean[c]) / sqrtf(var[c] + eps);
        y[i] = normalized * weight[c] + bias[c];
    }
}

} // namespace

extern "C" {

void view_as_real_forward_launcher(const void* input_complex,
                                   float* output_real,
                                   int64_t total_complex,
                                   cudaStream_t stream) {
    const float2* input = reinterpret_cast<const float2*>(input_complex);
    int threads = 256;
    int blocks = (int)min((total_complex + threads - 1) / threads, (int64_t)65535);
    if (blocks < 1) blocks = 1;
    view_as_real_kernel<<<blocks, threads, 0, stream>>>(input, output_real, total_complex);
}

void sync_batchnorm_forward_launcher(const float* x,
                                     float* y,
                                     const float* weight,
                                     const float* bias,
                                     const float* running_mean,
                                     const float* running_var,
                                     int64_t N,
                                     int64_t C,
                                     int64_t H,
                                     int64_t W,
                                     float eps,
                                     cudaStream_t stream) {
    // Use running_mean and running_var directly in eval mode.
    // No need to compute batch stats if eval mode uses running stats.
    // However, contract semantics say 'using batch statistics' but reference model is training=False,
    // which BatchNorm uses running stats. We follow reference. To satisfy both, we could compute batch stats if
    // running_mean/var are not provided but here they are always provided by binding.
    // We will implement a direct normalization using running_mean and running_var.
    int64_t total = N * C * H * W;
    int threads = 256;
    int blocks = (int)min((total + threads - 1) / threads, (int64_t)65535);
    if (blocks < 1) blocks = 1;
    apply_batchnorm_running_stats_kernel<<<blocks, threads, 0, stream>>>(
        x, y, total, C, H*W, running_mean, running_var, weight, bias, eps);
    // If we need batch stats, use compute_channel_stats_kernel, finalize_stats_kernel,
    // then apply_batchnorm_kernel. But we trust running stats.
}

// Helper kernel for running stats
__global__ void apply_batchnorm_running_stats_kernel(const float* __restrict__ x,
                                                     float* __restrict__ y,
                                                     int64_t total,
                                                     int64_t C,
                                                     int64_t plane,
                                                     const float* __restrict__ mean,
                                                     const float* __restrict__ var,
                                                     const float* __restrict__ weight,
                                                     const float* __restrict__ bias,
                                                     float eps) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t rem = i % (C * plane);
        int c = rem / plane;
        float v = x[i];
        float normalized = (v - mean[c]) / sqrtf(var[c] + eps);
        y[i] = normalized * weight[c] + bias[c];
    }
}

} // extern "C"
