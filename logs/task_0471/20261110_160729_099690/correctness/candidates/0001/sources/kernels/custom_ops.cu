// CUDA kernels for custom operations: view_as_real (complex64), log_softmax (dim=-1), group_norm (no affine)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper: convert complex64 (two consecutive floats real, imag) to real2
inline __device__ float2 complex64_to_float2(const float2& c) {
    return c;
}

// Kernel 1: view_as_real for complex64 -> float32 shape [..., 2]
// Input: complex64 tensor viewed as float2 array of length N (where N = num_complex_elements)
// Output: float32 tensor shape [..., 2] with same outer dimensions, i.e., 2*N floats total
// Each complex element (real, imag) is written to output[2*i] = real, output[2*i+1] = imag.
__global__ void view_as_real_kernel(const float2* __restrict__ input, float* __restrict__ output, long long n_complex) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < n_complex; i += stride) {
        float2 v = input[i];
        output[2 * i] = v.x;
        output[2 * i + 1] = v.y;
    }
}

// Kernel 2: log_softmax over last dimension of size 2
// Input and output are float arrays, each "row" has 2 elements (real, imag)
// total_rows = total number of such pairs = numel / 2
__global__ void log_softmax_dim_last_kernel(const float* __restrict__ input, float* __restrict__ output, long long total_rows) {
    long long row = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long r = row; r < total_rows; r += stride) {
        float a = input[2 * r];
        float b = input[2 * r + 1];
        float m = fmaxf(a, b);
        float sum_exp = __expf(a - m) + __expf(b - m);
        float log_sum = m + __logf(sum_exp);
        output[2 * r] = a - log_sum;
        output[2 * r + 1] = b - log_sum;
    }
}

// Kernel 3: group normalization without affine parameters
// Input shape: [N, C, D1, D2, D3] with D1=128, D2=128, D3=2 => per-channel spatial size = D1*D2*D3 = 32768
// Contract: N=512, C=8, num_groups=4 => channels_per_group = 2
// For each (n, g) pair, compute mean and variance over all elements in that group across all spatial dims.
// Then normalize: (x - mean) / sqrt(var + eps)
// We use a single block per (n,g) to compute statistics, then a second kernel to apply normalization.
// For performance, we can compute per-block reductions using shared memory, but with group size of 2 channels and spatial 32768, total elements per group = 65536.
// A block of 256 threads with grid-stride loop can reduce effectively.
// For simplicity and correctness, we implement a two-pass approach: first compute mean and variance using atomic or reduction; here we use a kernel that computes per-(n,g) sum and sum_sq using a grid of blocks, then a second kernel to apply normalization.
// However, to avoid multiple kernel launches and atomic overhead, we can use a single kernel with block reduction and shared memory to compute stats per group, then apply within same kernel? But applying normalization requires stats computed before normalization, so temporary storage is needed.
// Because the total number of groups is N * num_groups = 512*4=2048, and each group has 65536 elements, we can use one block per group with a grid-stride loop for reduction and then apply normalization in the same block after syncthreads, assuming we have enough shared memory for stats (just mean and var) and all threads can access them. Since each block handles one entire group, 256 threads can reduce 65536 elements using sequential accumulation per thread or warp reductions. We'll use a loop where each thread accumulates partial sums over its strided elements into registers, then do a block reduction via shared memory. Then after syncthreads, all threads have mean and var and can apply normalization on their assigned elements (same strided pattern). This avoids global synchronization and multiple kernels.
// Thus, launch one block per group. gridDim = N * num_groups, blockDim = 256.
// Each block processes elements in the group linearized as: group_index = n * num_groups + g, channels = g * channels_per_group + c_local (c_local in 0..channels_per_group-1), spatial index from 0 to D1*D2*D3-1.
// We need input offset: n * (C * spatial) + (g*channels_per_group + c_local) * spatial + spatial_idx
// For each thread, loop over total_elements_per_group = channels_per_group * spatial = 2 * 32768 = 65536 with stride blockDim.
// Accumulate sum and sum_sq in double precision for accuracy? Use float as contract expects float tolerance.
// After reduction, compute mean = sum / count, var = sum_sq/count - mean^2 (or use two-pass for numerical stability: variance = (sum_sq - sum^2/count)/count). Since count is large (65536), float is acceptable with tolerance 1e-2.

__global__ void group_norm_kernel(const float* __restrict__ input, float* __restrict__ output,
                                  int N, int C, int spatial_size, int num_groups, int channels_per_group, float eps) {
    int group_id = blockIdx.x; // 0 .. N*num_groups-1
    int n = group_id / num_groups;
    int g = group_id - n * num_groups;
    // total elements for this group
    int total = channels_per_group * spatial_size;
    // base pointer offset for this n, g, first channel
    const float* group_ptr = input + n * (C * spatial_size) + g * (channels_per_group * spatial_size);

    float thread_sum = 0.0f;
    float thread_sumsq = 0.0f;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        // idx maps to c_local = idx / spatial_size, spatial_idx = idx % spatial_size
        float val = group_ptr[idx];
        thread_sum += val;
        thread_sumsq += val * val;
    }

    // Block reduction for sum and sumsq
    __shared__ float shared_sum[256];
    __shared__ float shared_sumsq[256];
    int tid = threadIdx.x;
    shared_sum[tid] = thread_sum;
    shared_sumsq[tid] = thread_sumsq;
    __syncthreads();

    // Reduce within block using sequential addressing
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sumsq[tid] += shared_sumsq[tid + s];
        }
        __syncthreads();
    }

    float sum = shared_sum[0];
    float sumsq = shared_sumsq[0];
    float count = (float)total;
    float mean = sum / count;
    float var = sumsq / count - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv_std = rsqrtf(var + eps);

    // Apply normalization
    float* out_group_ptr = output + n * (C * spatial_size) + g * (channels_per_group * spatial_size);
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        float val = group_ptr[idx];
        out_group_ptr[idx] = (val - mean) * inv_std;
    }
}

// Launcher functions with extern "C" and stable signatures

extern "C" void view_as_real_complex64_launcher(const void* input, float* output, long long n_complex, cudaStream_t stream) {
    const int threads = 256;
    long long blocks = (n_complex + threads - 1) / threads;
    // grid cap? Use grid-stride loop in kernel, so can cap at 65535 blocks if needed but with grid-stride loop, we can use a larger grid if possible.
    // We'll ensure grid size doesn't exceed max grid dims by using a reasonable number: min(blocks, 65535)
    long long grid = blocks;
    if (grid > 65535) grid = 65535;
    view_as_real_kernel<<<(unsigned int)grid, threads, 0, stream>>>((const float2*)input, output, n_complex);
}

extern "C" void log_softmax_dim_last_launcher(const float* input, float* output, long long total_rows, cudaStream_t stream) {
    const int threads = 256;
    long long blocks = (total_rows + threads - 1) / threads;
    long long grid = blocks;
    if (grid > 65535) grid = 65535;
    log_softmax_dim_last_kernel<<<(unsigned int)grid, threads, 0, stream>>>(input, output, total_rows);
}

extern "C" void group_norm_fp32_launcher(const float* input, float* output, int N, int C, int spatial_size, int num_groups, float eps, cudaStream_t stream) {
    int channels_per_group = C / num_groups;
    int total_groups = N * num_groups;
    const int threads = 256;
    // one block per group
    group_norm_kernel<<<total_groups, threads, 0, stream>>>(input, output, N, C, spatial_size, num_groups, channels_per_group, eps);
}
