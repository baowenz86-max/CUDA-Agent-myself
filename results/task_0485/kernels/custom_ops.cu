#include <cuda_runtime.h>
#include <cmath>

// log_softmax kernel: computes log_softmax along the last dimension.
// Each block handles one row (last dimension size = inner_dim).
// gridDim.x = batch_size * other_dims; blockDim.x = inner_dim (up to 1024).
__global__ void log_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int inner_dim,
    int total_rows)
{
    extern __shared__ float smem[]; // will store the max value and sum
    int row = blockIdx.x;
    if (row >= total_rows) return;

    const float* input_row = input + row * inner_dim;
    float* output_row = output + row * inner_dim;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Step 1: find max for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < inner_dim; i += stride) {
        float v = input_row[i];
        if (v > max_val) max_val = v;
    }

    // block reduction for max
    __shared__ float sh_max[1024];
    sh_max[tid] = max_val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_max[tid] = fmaxf(sh_max[tid], sh_max[tid + offset]);
        }
        __syncthreads();
    }
    float block_max = sh_max[0];

    // Step 2: compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = tid; i < inner_dim; i += stride) {
        sum += __expf(input_row[i] - block_max);
    }

    // block reduction for sum
    __shared__ float sh_sum[1024];
    sh_sum[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_sum[tid] += sh_sum[tid + offset];
        }
        __syncthreads();
    }
    float block_sum = sh_sum[0];

    // Step 3: compute log_softmax = (x - max) - log(sum)
    float log_sum = logf(block_sum);
    for (int i = tid; i < inner_dim; i += stride) {
        output_row[i] = (input_row[i] - block_max) - log_sum;
    }
}

// tanh kernel: elementwise tanh, grid-stride loop
__global__ void tanh_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long n)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < n; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

// Launcher for log_softmax
// input, output: cuda pointers to float arrays; inner_dim: size of last dim; total_rows: number of rows
extern "C" void log_softmax_launcher(
    const float* input,
    float* output,
    int inner_dim,
    int total_rows,
    cudaStream_t stream)
{
    // Choose block size as next multiple of 32 <= inner_dim
    int block = 32;
    while (block < inner_dim && block < 1024) block <<= 1;
    if (block > inner_dim) block >>= 1;
    if (block == 0) block = 1;

    dim3 grid(total_rows);
    // shared memory size not needed beyond built-in static arrays in kernel
    log_softmax_kernel<<<grid, block, 0, stream>>>(
        input, output, inner_dim, total_rows);
}

// Launcher for tanh
// input, output: cuda pointers to float arrays; n: number of elements
extern "C" void tanh_launcher(
    const float* input,
    float* output,
    long long n,
    cudaStream_t stream)
{
    int block = 256;
    // calculate grid to cover all elements, but cap grid size
    long long grid_size = (n + block - 1) / block;
    if (grid_size > 65535) grid_size = 65535;
    tanh_kernel<<<dim3((unsigned int)grid_size), block, 0, stream>>>(
        input, output, n);
}
