#include <cuda_runtime.h>
#include <math.h>

// Softplus forward kernel: y = x > threshold ? x : log1p(exp(x))
// threshold = 20.0 for beta=1.0
__global__ void softplus_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        float x = input[i];
        // For large positive x, exp(x) may overflow; threshold avoids that.
        output[i] = (x > 20.0f) ? x : log1pf(expf(x));
    }
}

// InstanceNorm3d kernel: each block handles one (N,C) pair.
// spatial_size = D*H*W.
// We use Welford's algorithm for stable mean and variance accumulation,
// then a second loop to normalize.
template <int BLOCK_SIZE>
__global__ void instance_norm3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int spatial_size,
    float eps) {
    // blockIdx.x encodes (n, c) pair: n = blockIdx.x / C, c = blockIdx.x % C
    int n = blockIdx.x / C;
    int c = blockIdx.x - n * C;
    // Base offset for this (n,c) in the contiguous layout (N,C,D,H,W)
    // Since input and output are contiguous (we'll ensure in binding), offset = (n*C + c)*spatial_size
    long long base = ((long long)n * C + c) * spatial_size;

    // Shared memory for reduction: mean and M2
    __shared__ float s_mean[BLOCK_SIZE];
    __shared__ float s_m2[BLOCK_SIZE];

    int tid = threadIdx.x;
    int stride = BLOCK_SIZE;

    // Initialize thread-local accumulators
    float local_mean = 0.0f;
    float local_m2 = 0.0f;
    int local_count = 0;

    // First pass: compute mean and variance using Welford's algorithm
    for (int i = base + tid; i < base + spatial_size; i += stride) {
        float value = input[i];
        local_count++;
        float delta = value - local_mean;
        local_mean += delta / local_count;
        float delta2 = value - local_mean;
        local_m2 += delta * delta2;
    }

    // Store partial results in shared memory
    s_mean[tid] = local_mean;
    s_m2[tid] = local_m2;
    __syncthreads();

    // Reduction in shared memory (only valid for threads with count>0, but we can assume all have same count if spatial_size divisible by BLOCK_SIZE, not necessarily; we need to handle count mismatch)
    // To handle variable counts per thread, we need to also store counts.
    // Alternatively, we can ensure spatial_size is always divisible by BLOCK_SIZE? It is 204800, not necessarily divisible by 256 (it is 204800/256 = 800 exactly). So for BLOCK_SIZE=256, each thread processes exactly 800 elements. That simplifies: all threads have same count, and reduction over s_mean and s_m2 with all threads active.
    // For generality, we'll still handle variable counts by using a separate count array or by performing reduction assuming all active threads have full count? For robustness, we'll use a warp/block reduction that combines means and M2 based on counts, but since our spatial_size is divisible by BLOCK_SIZE for chosen block size (256), we can assume equal counts. However, to be safe, we can store counts and do a weighted Welford merge.
    // Given contract has fixed shape [16,50,512,20,20], spatial_size=204800, and we choose BLOCK_SIZE=256 => 800 iterations exactly. So we can assume equal counts. But to be robust for other shapes, we'll implement a merge that handles counts.
    // Let's store count in shared memory as well.
    // For simplicity and performance, we'll specialize with BLOCK_SIZE=256 and assume divisibility, but we'll include a runtime check fallback? Since only this shape is used, we can keep it simple and ensure correctness for this shape.
    // We'll implement a full Welford merge for correctness with arbitrary shapes, but it may be slower; however performance is still likely fine.
    // We'll store counts in shared memory too.
    __shared__ int s_count[BLOCK_SIZE];
    s_count[tid] = local_count;
    __syncthreads();

    // Reduce over s_count, s_mean, s_m2 using tree reduction (for simplicity, linear reduction by thread 0)
    // Since BLOCK_SIZE <= 1024, a single thread can reduce sequentially after syncthreads.
    if (tid == 0) {
        float total_mean = 0.0f;
        float total_m2 = 0.0f;
        int total_count = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            int cnt = s_count[i];
            if (cnt > 0) {
                float delta = s_mean[i] - total_mean;
                float new_count = total_count + cnt;
                total_m2 += s_m2[i] + delta * delta * total_count * cnt / new_count;
                total_mean += delta * cnt / new_count;
                total_count = new_count;
            }
        }
        // Now total_mean and total_m2 / total_count give mean and variance (biased)
        float mean = total_mean;
        float variance = total_m2 / total_count;
        // Store in shared memory for all threads
        s_mean[0] = mean;
        s_m2[0] = variance;
    }
    __syncthreads();

    float mean = s_mean[0];
    float variance = s_m2[0];
    float inv_std = rsqrtf(variance + eps);

    // Second pass: normalize
    for (int i = base + tid; i < base + spatial_size; i += stride) {
        output[i] = (input[i] - mean) * inv_std;
    }
}

// Launcher for softplus (extern "C" to avoid name mangling)
extern "C" void softplus_forward_launcher(
    const float* input,
    float* output,
    int numel,
    cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding limits; use grid-stride loop if necessary.
    // For numel=163,840,000, grid_size=640,000 which may exceed max grid x (2^31-1) is fine, but performance may degrade due to launch overhead. We can use a reasonable grid size, e.g., 1024 blocks per dimension, but since kernel uses grid-stride loop, any grid size works. We'll choose a grid that saturates GPU: e.g., number of SMs * some factor.
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int max_blocks_per_sm;
    // Simple heuristic: use 4 times number of SMs, but cap.
    int num_sms = prop.multiProcessorCount;
    int desired_blocks = num_sms * 4;
    if (desired_blocks > grid_size) desired_blocks = grid_size;
    if (desired_blocks < 1) desired_blocks = 1;
    softplus_forward_kernel<<<desired_blocks, block_size, 0, stream>>>(input, output, numel);
}

// Launcher for instance_norm3d
extern "C" void instance_norm3d_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int spatial_size,
    float eps,
    cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = N * C;
    // Ensure block_size divides spatial_size if we rely on equal counts; otherwise, we would need the merge path,
    // but our merge handles variable counts correctly, so it's fine.
    instance_norm3d_kernel<256><<<num_blocks, block_size, 0, stream>>>(
        input, output, N, C, spatial_size, eps);
}
