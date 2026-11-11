#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Instance Normalization (InstanceNorm2d with affine=False, track_running_stats=False)
// For each (n, c) pair, compute mean and biased variance over H*W spatial elements,
// then normalize: (x - mean) / sqrt(var + eps)
// -----------------------------------------------------------------------------

// Block size for instance norm reduction
#define INSTANCE_NORM_BLOCK_SIZE 256

__global__ void instance_norm2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int HW,
    float eps
) {
    // Each block handles one (n, c) channel
    int nc = blockIdx.x;                  // combined n*C + c
    int n = nc / C;
    int c = nc % C;

    const float* input_channel = input + (size_t)nc * HW;
    float* output_channel = output + (size_t)nc * HW;

    // Shared memory for block reduction
    __shared__ float sdata[INSTANCE_NORM_BLOCK_SIZE];

    // Thread index
    int tid = threadIdx.x;

    // Compute sum and sum of squares using grid-stride loop within the block
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid; i < HW; i += blockDim.x) {
        float v = input_channel[i];
        sum += v;
        sum_sq += v * v;
    }

    // Block reduction for sum and sum_sq
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float block_sum = sdata[0];

    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float block_sum_sq = sdata[0];

    // Compute mean and variance (biased)
    float mean = block_sum / HW;
    float var = block_sum_sq / HW - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Normalize all elements in this channel
    for (int i = tid; i < HW; i += blockDim.x) {
        output_channel[i] = (input_channel[i] - mean) * inv_std;
    }
}

// Launcher for instance_norm2d_forward
extern "C" void instance_norm2d_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    float eps,
    cudaStream_t stream
) {
    int HW = H * W;
    int NC = N * C;

    // One block per (n, c), block size = INSTANCE_NORM_BLOCK_SIZE
    dim3 grid(NC);
    dim3 block(INSTANCE_NORM_BLOCK_SIZE);

    instance_norm2d_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, HW, eps
    );

    // Kernel launch error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, this should be surfaced as an exception.
        // Here we just return; the binding layer can check cudaGetLastError if needed.
        return;
    }
}

// -----------------------------------------------------------------------------
// LogSoftmax along dim=1 (channel dimension)
// For each (n, h, w) position, compute log_softmax over C channels:
//   y_i = x_i - max_j(x_j) - log(sum_j exp(x_j - max_j(x_j)))
// -----------------------------------------------------------------------------

// Block size for logsoftmax: one block per spatial position
#define LOGSOFTMAX_BLOCK_SIZE 256

__global__ void log_softmax_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int HW
) {
    // Each block handles one spatial position (n, h, w), i.e., one column over channels
    int spatial_idx = blockIdx.x;   // index in [0, N*HW)
    int n = spatial_idx / HW;
    int hw = spatial_idx % HW;

    // Base offset for this spatial position across channels
    size_t base = (size_t)n * C * HW + hw;
    const float* input_pos = input + base;
    float* output_pos = output + base;

    // Shared memory for reduction across channels (up to LOGSOFTMAX_BLOCK_SIZE threads)
    __shared__ float sdata[LOGSOFTMAX_BLOCK_SIZE];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Compute max over channels using grid-stride loop (C may be larger than blockDim.x)
    float max_val = -INFINITY;
    for (int c = tid; c < C; c += num_threads) {
        float v = input_pos[(size_t)c * HW];
        if (v > max_val) max_val = v;
    }

    // Block reduction for max
    sdata[tid] = max_val;
    __syncthreads();
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = sdata[tid + s];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }
    float block_max = sdata[0];

    // Compute sum of exp(x - max) over channels
    float sum_exp = 0.0f;
    for (int c = tid; c < C; c += num_threads) {
        float v = input_pos[(size_t)c * HW];
        sum_exp += __expf(v - block_max);
    }

    // Block reduction for sum
    sdata[tid] = sum_exp;
    __syncthreads();
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float block_sum = sdata[0];
    float log_sum = logf(block_sum);

    // Write output: x - max - log(sum_exp)
    for (int c = tid; c < C; c += num_threads) {
        float v = input_pos[(size_t)c * HW];
        output_pos[(size_t)c * HW] = v - block_max - log_sum;
    }
}

// Launcher for log_softmax_dim1_forward
extern "C" void log_softmax_dim1_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    cudaStream_t stream
) {
    int HW = H * W;
    int total_spatial = N * HW;

    // Choose block size; if C is small we can use fewer threads, but keep simple
    int block_size = LOGSOFTMAX_BLOCK_SIZE;
    // Limit block size to at least C if C < block_size; but must be multiple of 32 for efficiency
    // For safety, if C < block_size, we can use a smaller multiple of 32
    if (C < block_size) {
        int candidate = 32;
        while (candidate < C) candidate <<= 1;
        block_size = candidate;
    }

    dim3 grid(total_spatial);
    dim3 block(block_size);

    log_softmax_dim1_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, HW
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return;
    }
}
