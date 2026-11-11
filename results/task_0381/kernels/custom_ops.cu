#include <cuda_runtime.h>
#include <float.h>

// -----------------------------------------------------------------------------
// Kernel 1: custom_hardtanh
// Applies hardtanh activation: output = min(max(x, min_val), max_val)
// Supports arbitrary strides, but for performance assumes contiguous input
// (the binding should provide contiguous or handle stride via copy).
// -----------------------------------------------------------------------------
__global__ void hardtanh_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    float min_val,
    float max_val
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        val = fminf(fmaxf(val, min_val), max_val);
        output[idx] = val;
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: custom_slice_last_dim
// Input shape: [B, C, D, H, W, 1]
// Output shape: [B, C, D, H, W]
// Copies the element at last dimension index 0 for each output position.
// Input is expected contiguous; output contiguous.
// -----------------------------------------------------------------------------
__global__ void slice_last_dim_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t output_numel
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    // Since last dimension size is 1, input index equals output index.
    for (; idx < output_numel; idx += stride) {
        output[idx] = input[idx];
    }
}

// -----------------------------------------------------------------------------
// Fused instance norm kernel: one block per batch.
// Each block computes mean and variance of the batch, then normalizes all elements.
// -----------------------------------------------------------------------------
__global__ void instance_norm_fused_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t spatial_numel,
    float eps
) {
    extern __shared__ float smem[]; // first half sum, second half sumsq
    float* s_sum = smem;
    float* s_sumsq = smem + blockDim.x;

    int batch = blockIdx.x;
    const float* batch_in = input + batch * spatial_numel;
    float* batch_out = output + batch * spatial_numel;

    // Each thread accumulates partial sums
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    int64_t stride = blockDim.x;
    for (int64_t i = threadIdx.x; i < spatial_numel; i += stride) {
        float val = batch_in[i];
        local_sum += val;
        local_sumsq += val * val;
    }

    s_sum[threadIdx.x] = local_sum;
    s_sumsq[threadIdx.x] = local_sumsq;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Broadcast mean and variance
    __shared__ float mean_val;
    __shared__ float var_val;
    if (threadIdx.x == 0) {
        float total_sum = s_sum[0];
        float total_sumsq = s_sumsq[0];
        mean_val = total_sum / spatial_numel;
        var_val = fmaxf(total_sumsq / spatial_numel - mean_val * mean_val, 0.0f);
    }
    __syncthreads();

    float m = mean_val;
    float v = var_val;
    float inv_std = rsqrtf(v + eps);

    // Normalize all elements
    for (int64_t i = threadIdx.x; i < spatial_numel; i += stride) {
        batch_out[i] = (batch_in[i] - m) * inv_std;
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C" to be callable from binding)
// -----------------------------------------------------------------------------
extern "C" void custom_hardtanh_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    // Limit blocks to reasonable maximum (e.g. 65535 for old grids), but use grid-stride loop
    if (blocks > 65535) blocks = 65535;
    hardtanh_kernel<<<blocks, threads, 0, stream>>>(output, input, numel, min_val, max_val);
}

extern "C" void custom_slice_last_dim_launcher(
    float* output,
    const float* input,
    int64_t output_numel,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (int)((output_numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    slice_last_dim_kernel<<<blocks, threads, 0, stream>>>(output, input, output_numel);
}

extern "C" void custom_instance_norm_3d_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t channels,
    int64_t depth,
    int64_t height,
    int64_t width,
    double eps,
    cudaStream_t stream
) {
    // In this task, channels == 1
    int64_t spatial_numel = depth * height * width;
    // total_numel = batch_size * channels * spatial_numel; // = batch_size * spatial_numel for C=1

    int threads = 256;
    int blocks = (int)batch_size; // one block per batch

    dim3 grid(blocks);
    dim3 block(threads);
    size_t shared_mem_size = 2 * threads * sizeof(float); // for sum and sumsq reduction
    instance_norm_fused_kernel<<<grid, block, shared_mem_size, stream>>>(
        output, input, spatial_numel, (float)eps
    );
}
