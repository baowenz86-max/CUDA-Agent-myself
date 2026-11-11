#include <cuda_runtime.h>
#include <float.h>

// -----------------------------------------------------------------------------
// Utility: get number of blocks for grid-stride loops
// -----------------------------------------------------------------------------
__device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

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
// Kernel 3: custom_instance_norm_3d
// Input shape: [B, C, D, H, W], with C=1 in this task.
// Performs instance normalization per batch and channel over spatial dims D,H,W.
// Since C=1, each batch's entire spatial volume is normalized.
// We use a two-pass approach: first compute mean and variance, then normalize.
// For large spatial sizes, a single block per batch is used to ensure sequential consistency.
// -----------------------------------------------------------------------------

// First pass: compute per-batch mean and variance.
// Uses one block per batch. Each block reduces over spatial elements.
// Assumes spatial_numel = D*H*W, input is contiguous.
__global__ void instance_norm_stats_kernel(
    const float* __restrict__ input,
    float* __restrict__ mean,
    float* __restrict__ var,
    int64_t spatial_numel
) {
    extern __shared__ float smem[]; // [0..blockDim.x-1] for reduction
    int batch = blockIdx.x;
    const float* batch_ptr = input + batch * spatial_numel;

    // Each thread computes local sum and sum of squares
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    int64_t stride = blockDim.x;
    for (int64_t i = threadIdx.x; i < spatial_numel; i += stride) {
        float val = batch_ptr[i];
        local_sum += val;
        local_sumsq += val * val;
    }

    // Parallel reduction within block
    // Use shared memory for sums
    __shared__ float sum_shared[256];
    __shared__ float sumsq_shared[256];
    if (threadIdx.x < 256) {
        sum_shared[threadIdx.x] = local_sum;
        sumsq_shared[threadIdx.x] = local_sumsq;
    }
    __syncthreads();

    // Reduce if blockDim.x > 256 not possible (we will launch 256 threads per block)
    // Here we assume blockDim.x = 256
    if (blockDim.x == 256) {
        for (int s = 128; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sum_shared[threadIdx.x] += sum_shared[threadIdx.x + s];
                sumsq_shared[threadIdx.x] += sumsq_shared[threadIdx.x + s];
            }
            __syncthreads();
        }
    } else {
        // generic reduction (fallback)
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sum_shared[threadIdx.x] += sum_shared[threadIdx.x + s];
                sumsq_shared[threadIdx.x] += sumsq_shared[threadIdx.x + s];
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        float total_sum = sum_shared[0];
        float total_sumsq = sumsq_shared[0];
        float mean_val = total_sum / spatial_numel;
        float var_val = fmaxf(total_sumsq / spatial_numel - mean_val * mean_val, 0.0f);
        mean[batch] = mean_val;
        var[batch] = var_val;
    }
}

// Second pass: normalize using stored mean and variance
__global__ void instance_norm_normalize_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    int64_t spatial_numel
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t total_elements = (int64_t)gridDim.x * blockDim.x; // not used
    // Each block handles a range within global input, but we need batch index.
    // Since C=1, batch index = idx / spatial_numel.
    // For each element, compute batch and local index.
    for (; idx < (int64_t)blockDim.x * gridDim.x * 0 + (int64_t)gridDim.x * blockDim.x; idx += stride) {
        // This loop is replaced below for clarity; we will use a single pass.
        break;
    }
    // Proper implementation: grid-stride over all elements, identify batch.
    int64_t total = (int64_t)gridDim.x * blockDim.x; // dummy
    // Actually launch with enough blocks; here we compute per block batches.
    // We'll rewrite: each thread iterates over its assigned elements.
}

// Combined normalization kernel (avoids separate second kernel by computing per element)
__global__ void instance_norm_apply_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    int64_t spatial_numel,
    int64_t total_numel
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total_numel; idx += stride) {
        int batch = idx / spatial_numel;
        float m = mean[batch];
        float v = var[batch];
        float inv_std = rsqrtf(v + eps);
        output[idx] = (input[idx] - m) * inv_std;
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
    int blocks = (int)ceil_div(numel, threads);
    // Limit blocks to reasonable maximum (e.g. 65535 for old grids, but we can use grid-stride)
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
    int blocks = (int)ceil_div(output_numel, threads);
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
    int64_t total_numel = batch_size * channels * spatial_numel; // = batch_size * spatial_numel

    // Allocate temporary memory for mean and variance (batch_size floats each)
    // We'll use cudaMallocAsync or simple cudaMalloc; but launcher should avoid repeated allocation.
    // Simpler: use a single kernel that computes stats then normalizes, using persistent block per batch.
    // We'll implement a fused kernel with shared memory reduction and normalization in one launch.
    // This avoids intermediate allocations and extra synchronization.
    int threads = 256;
    int blocks = batch_size; // one block per batch

    // Fused kernel: each block handles one batch, computes mean/var in shared memory,
    // then normalizes all elements of that batch.
    dim3 grid(blocks);
    dim3 block(threads);
    size_t shared_mem_size = 2 * threads * sizeof(float); // for sum and sumsq reduction
    instance_norm_fused_kernel<<<grid, block, shared_mem_size, stream>>>(
        output, input, spatial_numel, (float)eps
    );
}

// Fused instance norm kernel: one block per batch.
// Each block computes mean and variance of the batch, then normalizes all elements.
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
