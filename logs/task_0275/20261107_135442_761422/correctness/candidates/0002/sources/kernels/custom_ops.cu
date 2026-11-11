#include <cuda_runtime.h>
#include <cmath>

// Normalize kernel (used in both modes with appropriate mean/var pointers)
__global__ void batchnorm2d_normalize_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = N * C * H * W;
    for (int i = idx; i < total; i += stride) {
        int n = i / (C * H * W);
        int rem = i - n * (C * H * W);
        int c = rem / (H * W);
        int hw = rem - c * (H * W);
        int h = hw / W;
        int w = hw - h * W;
        long long in_idx = n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w;
        float val = input[in_idx];
        float m = mean[c];
        float v = var[c];
        float invstd = rsqrtf(v + eps);
        float result = (val - m) * invstd;
        if (weight) result *= weight[c];
        if (bias) result += bias[c];
        long long out_idx = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
        output[out_idx] = result;
    }
}

// Custom Batchnorm2d kernel
// Computes output = (input - mean) * invstd * weight + bias
// For training mode, mean and var are computed over batch and spatial dims per channel,
// and running_mean/running_var are updated (in-place).
// For eval mode, uses provided running_mean/running_var.
// Input and output are 4D tensors with shape [N,C,H,W], but we treat them as 1D arrays
// with associated strides to handle non-contiguous inputs.
__global__ void batchnorm2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ running_mean_out,   // may be same as running_mean
    float* __restrict__ running_var_out,    // may be same as running_var
    float eps,
    bool training,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float momentum,
    float inv_momentum, // 1 - momentum for running updates
    int total_spatial // H*W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N*C*H*W; i += stride) {
        int n = i / (C * H * W);
        int rem = i - n * (C * H * W);
        int c = rem / (H * W);
        int hw = rem - c * (H * W);
        int h = hw / W;
        int w = hw - h * W;

        long long in_idx = n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w;
        float val = input[in_idx];

        float mean, var;
        if (training) {
            // In training, mean and var are computed per channel per batch; this requires reduction.
            // For simplicity here, we assume the training path is not exercised; but we implement a
            // naive two-pass approach: first pass computes mean/var, second pass normalizes.
            // This kernel is only called after the reduction has produced channel statistics,
            // which we store in running_mean/running_var temporarily.
            // However, we need a separate mechanism. Since training is not in the reference, we leave this
            // path but indicate it requires precomputed stats.
            // To avoid complex synchronization, we will implement training using atomic accumulation in a separate kernel,
            // but here we assume mean/var already available in running_mean/running_var.
            mean = running_mean[c];
            var = running_var[c];
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }

        float invstd = rsqrtf(var + eps);
        float normalized = (val - mean) * invstd;
        if (weight != nullptr) normalized *= weight[c];
        if (bias != nullptr) normalized += bias[c];

        long long out_idx = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
        output[out_idx] = normalized;
    }
}

// Kernel to compute batch statistics (mean, var) per channel in training mode.
// Uses two passes within a single kernel with shared memory reduction per block, then atomics.
// We won't call this in eval mode.
__global__ void batchnorm2d_compute_stats_kernel(
    const float* __restrict__ input,
    float* __restrict__ mean,
    float* __restrict__ var,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    int total_spatial // H*W
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int c = blockIdx.x;
    if (c >= C) return;

    // Each block computes mean and var for one channel across all N and spatial positions.
    float sum = 0.f;
    float sumsq = 0.f;
    int count = N * total_spatial;
    for (int i = tid; i < count; i += blockDim.x) {
        int n = i / total_spatial;
        int hw = i - n * total_spatial;
        int h = hw / W;
        int w = hw - h * W;
        long long in_idx = n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w;
        float val = input[in_idx];
        sum += val;
        sumsq += val * val;
    }

    // Block reduction to shared memory (simple tree)
    // We'll use sdata[0] for sum, sdata[1] for sumsq
    sdata[tid] = sum;
    sdata[blockDim.x + tid] = sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float total_sum = sdata[0];
        float total_sumsq = sdata[blockDim.x];
        float batch_mean = total_sum / count;
        float batch_var = total_sumsq / count - batch_mean * batch_mean;
        mean[c] = batch_mean;
        var[c] = batch_var;
    }
}

// Kernel to update running stats in training mode (called after compute stats)
__global__ void batchnorm2d_update_running_kernel(
    float* running_mean,
    float* running_var,
    const float* batch_mean,
    const float* batch_var,
    int C,
    float momentum,
    float inv_momentum
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        running_mean[c] = (1.f - momentum) * running_mean[c] + momentum * batch_mean[c];
        running_var[c] = (1.f - momentum) * running_var[c] + momentum * batch_var[c];
    }
}

// GELU kernel (approximate='none'): out = x * 0.5 * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel
) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < numel) {
        float x = input[i];
        output[i] = x * 0.5f * (1.0f + erff(x * 0.70710678118654752440f));
    }
}

// Launcher for custom_batchnorm2d
// All pointers are assumed to be valid CUDA device pointers.
// weight and bias may be null.
// running_mean/running_var are input for eval; for training they serve as input for current running stats,
// and are updated in-place.
// Output is a newly allocated tensor with same strides as input? Actually output should be contiguous
// according to contract, but we match input's shape; we assume output is contiguous for simplicity.
// To handle non-contiguous output, we pass output strides as parameters; but the binding will allocate contiguous output.
// The launcher computes grid/block based on total elements.
extern "C" void custom_batchnorm2d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    float* running_mean,
    float* running_var,
    float eps,
    bool training,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float momentum,
    cudaStream_t stream
) {
    int total_elements = N * C * H * W;
    if (total_elements == 0) return;

    if (training) {
        // Compute batch statistics
        int block = 256;
        int grid_stats = C; // one block per channel
        size_t shmem = 2 * block * sizeof(float);
        // Need temporary storage for mean and var per channel
        float *batch_mean, *batch_var;
        cudaMalloc(&batch_mean, C * sizeof(float));
        cudaMalloc(&batch_var, C * sizeof(float));
        if (!batch_mean || !batch_var) return; // error handling skipped

        batchnorm2d_compute_stats_kernel<<<grid_stats, block, shmem, stream>>>(
            input, batch_mean, batch_var, N, C, H, W,
            in_stride_n, in_stride_c, in_stride_h, in_stride_w,
            H * W
        );

        // Update running stats (in-place on running_mean/running_var)
        int grid_update = (C + 255) / 256;
        batchnorm2d_update_running_kernel<<<grid_update, 256, 0, stream>>>(
            running_mean, running_var, batch_mean, batch_var, C, momentum, 1.f - momentum
        );

        // Now normalize using batch statistics (we need to use batch_mean/batch_var, not running)
        int grid_norm = (total_elements + 255) / 256;
        batchnorm2d_normalize_kernel<<<grid_norm, 256, 0, stream>>>(
            output, input, weight, bias, batch_mean, batch_var, eps,
            N, C, H, W,
            in_stride_n, in_stride_c, in_stride_h, in_stride_w,
            out_stride_n, out_stride_c, out_stride_h, out_stride_w
        );

        cudaFree(batch_mean);
        cudaFree(batch_var);
    } else {
        // Eval mode: use running_mean/running_var
        int grid_norm = (total_elements + 255) / 256;
        batchnorm2d_normalize_kernel<<<grid_norm, 256, 0, stream>>>(
            output, input, weight, bias, running_mean, running_var, eps,
            N, C, H, W,
            in_stride_n, in_stride_c, in_stride_h, in_stride_w,
            out_stride_n, out_stride_c, out_stride_h, out_stride_w
        );
    }
}

// Launcher for custom_gelu
extern "C" void custom_gelu_launcher(
    float* output,
    const float* input,
    long long numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    int threads = 256;
    long long blocks = (numel + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}
