#include <cuda_runtime.h>
#include <math.h>

// LogSoftmax along dim=1 for 3D tensor with shape (B, H, W), contiguous memory.
// Each block handles one (b,w) vector of length H.
// Uses max subtraction for numerical stability.
__global__ void log_softmax_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B,            // batch size
    int H,            // height (dim=1)
    int W,            // width
    float* __restrict__ temp_max, // optional temp storage for max, but not needed if per-block reduce
    int use_shared_memory // not used, keep for compatibility placeholder
) {
    // blockIdx.x = batch index b (0..B-1)
    // blockIdx.y = width index w (0..W-1)
    int b = blockIdx.x;
    int w = blockIdx.y;
    if (b >= B || w >= W) return;

    int base = b * H * W + w;
    // Input is contiguous with stride: input[b][h][w] = input[base + h*W]
    // Output similarly.

    // Each thread handles a subset of H elements.
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // First pass: find maximum over H
    float max_val = -INFINITY;
    for (int h = tid; h < H; h += num_threads) {
        float val = input[base + h * W];
        if (val > max_val) max_val = val;
    }

    // Reduce max within block using shared memory
    extern __shared__ float smem[]; // smem[0] = max, smem[1] = sum
    // We need two stages: reduce max, then reduce sum.
    // Use two shared arrays or reuse after sync.
    // smem size: 2 * blockDim.x floats for partial reductions.
    float* s_max = smem;
    float* s_sum = smem + blockDim.x;

    s_max[tid] = max_val;
    __syncthreads();

    // Tree reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = s_max[tid + stride];
            if (other > s_max[tid]) s_max[tid] = other;
        }
        __syncthreads();
    }
    max_val = s_max[0];

    // Second pass: compute sum of exp(x - max)
    float sum = 0.0f;
    for (int h = tid; h < H; h += num_threads) {
        sum += __expf(input[base + h * W] - max_val);
    }

    s_sum[tid] = sum;
    __syncthreads();

    // Tree reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float total_sum = s_sum[0];
    float log_sum = logf(total_sum);

    // Third pass: write output = input - max - log_sum
    for (int h = tid; h < H; h += num_threads) {
        output[base + h * W] = input[base + h * W] - max_val - log_sum;
    }
}

// Simple elementwise multiply by scalar
__global__ void mul_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scalar,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        output[i] = input[i] * scalar;
    }
}

// Launcher for custom_log_softmax: input/output are contiguous float32 tensors of shape (B,H,W).
// stream is the CUDA stream to use.
extern "C" void custom_log_softmax_launcher(
    const float* input,
    float* output,
    int B,
    int H,
    int W,
    cudaStream_t stream
) {
    // Block size: use 256 threads, but adjust if H smaller? H is 1536, divisible by 256 (6 loops).
    // Could use blockDim = min(256, next_power_of_2(H)) for generality.
    int block_size = 256;
    if (H < block_size) {
        block_size = 1;
        while (block_size < H) block_size <<= 1;
    }
    dim3 grid(B, W);
    // Shared memory: two arrays of block_size floats
    int shared_mem = 2 * block_size * sizeof(float);
    log_softmax_dim1_kernel<<<grid, block_size, shared_mem, stream>>>(
        input, output, B, H, W, nullptr, 0);
}

// Launcher for custom_mul_scalar
extern "C" void custom_mul_scalar_launcher(
    const float* input,
    float* output,
    float scalar,
    int total_elements,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    // Cap grid to avoid excessive blocks? grid-stride loop handles any grid size.
    // Limit grid to some reasonable number to reduce overhead, e.g. 65535.
    if (grid_size > 65535) grid_size = 65535;
    mul_scalar_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, scalar, total_elements);
}
