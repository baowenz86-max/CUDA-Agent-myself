#include <cuda_runtime.h>
#include <cstdint>

// BatchNorm2d inference: y = (x - mean) / sqrt(var + eps) * weight + bias
__global__ void batch_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t HW)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * HW) return;
    int64_t c = (idx / HW) % C;
    float inv_std = rsqrtf(running_var[c] + eps);
    float scale = weight[c] * inv_std;
    float shift = bias[c] - running_mean[c] * scale;
    output[idx] = input[idx] * scale + shift;
}

// GroupNorm without affine: normalize each group in each sample
__global__ void group_norm_kernel_two_pass(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t HW,
    int64_t groups,
    float eps,
    float* __restrict__ group_mean,
    float* __restrict__ group_var)
{
    int64_t group_id = blockIdx.x;
    int64_t group_size = (C / groups) * HW;
    int64_t sample_id = blockIdx.y;
    if (group_id >= groups || sample_id >= N) return;

    // First pass: compute mean and variance within group
    float sum = 0.0f;
    float sq_sum = 0.0f;
    int64_t base = sample_id * C * HW + group_id * group_size;
    for (int64_t i = 0; i < group_size; ++i) {
        float val = input[base + i];
        sum += val;
        sq_sum += val * val;
    }
    float mean = sum / group_size;
    float var = sq_sum / group_size - mean * mean;
    group_mean[group_id * N + sample_id] = mean;
    group_var[group_id * N + sample_id] = var;

    __syncthreads();

    // Second pass: normalize
    float inv_std = rsqrtf(var + eps);
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        output[base + i] = (input[base + i] - mean) * inv_std;
    }
}

__global__ void group_norm_kernel_single_pass(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t HW,
    int64_t groups,
    float eps)
{
    extern __shared__ float sdata[];
    float* s_mean = sdata;
    float* s_var = sdata + blockDim.x;

    int64_t group_id = blockIdx.x;
    int64_t sample_id = blockIdx.y;
    int64_t group_size = (C / groups) * HW;
    int64_t base = sample_id * C * HW + group_id * group_size;

    // Compute partial sums per thread
    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        float val = input[base + i];
        sum += val;
        sq_sum += val * val;
    }

    // Block reduction
    s_mean[threadIdx.x] = sum;
    s_var[threadIdx.x] = sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_mean[threadIdx.x] += s_mean[threadIdx.x + stride];
            s_var[threadIdx.x] += s_var[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = s_mean[0] / group_size;
    float var = s_var[0] / group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        output[base + i] = (input[base + i] - mean) * inv_std;
    }
}

__global__ void histc_kernel(
    const float* __restrict__ input,
    int64_t numel,
    float min,
    float max,
    int64_t bins,
    float* __restrict__ output)
{
    __shared__ int local_hist[256];
    int tid = threadIdx.x;
    if (tid < bins) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + tid;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        if (val >= min && val <= max) {
            int64_t bin = (int64_t)((val - min) * bins / (max - min));
            if (bin >= bins) bin = bins - 1;
            atomicAdd(&local_hist[bin], 1);
        }
    }
    __syncthreads();

    if (tid < bins) {
        atomicAdd(&output[tid], (float)local_hist[tid]);
    }
}

// Launchers with C linkage to match binding declarations
extern "C" void batch_norm_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t HW,
    cudaStream_t stream)
{
    int64_t total = N * C * HW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    batch_norm_kernel<<<blocks, threads, 0, stream>>>(
        input, output, weight, bias, running_mean, running_var, eps, N, C, HW);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handling; binding will check after launch
    }
}

extern "C" void group_norm_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t HW,
    int64_t groups,
    float eps,
    cudaStream_t stream)
{
    int64_t group_size = (C / groups) * HW;
    int threads = 256;
    if (group_size <= threads) {
        // single pass with shared memory reduction
        dim3 grid((unsigned int)groups, (unsigned int)N);
        size_t shmem = 2 * threads * sizeof(float);
        group_norm_kernel_single_pass<<<grid, threads, shmem, stream>>>(
            input, output, N, C, HW, groups, eps);
    } else {
        // two pass, no shared memory arrays needed for reduction
        float* group_mean;
        float* group_var;
        cudaMalloc(&group_mean, groups * N * sizeof(float));
        cudaMalloc(&group_var, groups * N * sizeof(float));
        dim3 grid((unsigned int)groups, (unsigned int)N);
        group_norm_kernel_two_pass<<<grid, threads, 0, stream>>>(
            input, output, N, C, HW, groups, eps, group_mean, group_var);
        cudaFree(group_mean);
        cudaFree(group_var);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handling; binding will check after launch
    }
}

extern "C" void histc_launcher(
    const float* input,
    int64_t numel,
    float min,
    float max,
    int64_t bins,
    float* output,
    cudaStream_t stream)
{
    // reset output
    cudaMemsetAsync(output, 0, bins * sizeof(float), stream);
    int threads = 256;
    int blocks = 32;
    histc_kernel<<<blocks, threads, 0, stream>>>(
        input, numel, min, max, bins, output);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handling; binding will check after launch
    }
}
