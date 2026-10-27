#include <cuda_runtime.h>
#include <math_constants.h>

// Helper: ceil division
__device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// log_softmax along dim=1 for float32, NCHW layout, C=3 assumed but dynamic
// Each block handles one (n, spatial_index), threads across C
// Grid: N * H * W blocks, blockDim = C_threads (power of 2 >= C, up to 1024)
// For C=3 we use blockDim=32 for simplicity; dynamic C supported via shared memory reduction.
#define MAX_C 32

template <int BLOCK_SIZE>
__global__ void log_softmax_dim1_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int total_spatial)
{
    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_sum = smem + BLOCK_SIZE;

    int spatial_idx = blockIdx.x;
    if (spatial_idx >= total_spatial) return;

    int tid = threadIdx.x;
    int n = spatial_idx / (H * W);
    int hw = spatial_idx % (H * W);

    float local_max = -CUDART_INF_F;
    for (int c = tid; c < C; c += BLOCK_SIZE) {
        int idx = n * C * H * W + c * H * W + hw;
        float val = input[idx];
        local_max = fmaxf(local_max, val);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    float block_max = s_max[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int c = tid; c < C; c += BLOCK_SIZE) {
        int idx = n * C * H * W + c * H * W + hw;
        float val = input[idx];
        local_sum += __expf(val - block_max);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    float block_sum = s_sum[0];
    __syncthreads();

    float log_sum = logf(block_sum);

    for (int c = tid; c < C; c += BLOCK_SIZE) {
        int idx = n * C * H * W + c * H * W + hw;
        float val = input[idx];
        output[idx] = val - block_max - log_sum;
    }
}

template <int BLOCK_SIZE>
__global__ void softmax_dim1_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int total_spatial)
{
    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_sum = smem + BLOCK_SIZE;

    int spatial_idx = blockIdx.x;
    if (spatial_idx >= total_spatial) return;

    int tid = threadIdx.x;
    int n = spatial_idx / (H * W);
    int hw = spatial_idx % (H * W);

    float local_max = -CUDART_INF_F;
    for (int c = tid; c < C; c += BLOCK_SIZE) {
        int idx = n * C * H * W + c * H * W + hw;
        float val = input[idx];
        local_max = fmaxf(local_max, val);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    float block_max = s_max[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int c = tid; c < C; c += BLOCK_SIZE) {
        int idx = n * C * H * W + c * H * W + hw;
        float val = input[idx];
        local_sum += __expf(val - block_max);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    float block_sum = s_sum[0];
    __syncthreads();

    for (int c = tid; c < C; c += BLOCK_SIZE) {
        int idx = n * C * H * W + c * H * W + hw;
        float val = input[idx];
        output[idx] = __expf(val - block_max) / block_sum;
    }
}

__global__ void reflection_pad2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    int src_h = h - pad_h;
    if (src_h < 0) src_h = -src_h;
    if (src_h >= H_in) src_h = 2 * H_in - 2 - src_h;

    int src_w = w - pad_w;
    if (src_w < 0) src_w = -src_w;
    if (src_w >= W_in) src_w = 2 * W_in - 2 - src_w;

    while (src_h < 0 || src_h >= H_in) {
        if (src_h < 0) src_h = -src_h;
        else if (src_h >= H_in) src_h = 2 * H_in - 2 - src_h;
    }
    while (src_w < 0 || src_w >= W_in) {
        if (src_w < 0) src_w = -src_w;
        else if (src_w >= W_in) src_w = 2 * W_in - 2 - src_w;
    }

    int in_idx = ((n * C + c) * H_in + src_h) * W_in + src_w;
    output[idx] = input[in_idx];
}

extern "C" void log_softmax_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int dim,
    cudaStream_t stream)
{
    if (dim != 1) return;
    int total_spatial = N * H * W;
    int block_size = 32;
    while (block_size < C) block_size <<= 1;
    if (block_size > 256) block_size = 256;

    int shared_mem_size = 2 * block_size * sizeof(float);

    dim3 grid(total_spatial);
    dim3 block(block_size);

    switch(block_size) {
        case 32:
            log_softmax_dim1_kernel<32><<<grid, block, shared_mem_size, stream>>>(
                output, input, N, C, H, W, total_spatial);
            break;
        case 64:
            log_softmax_dim1_kernel<64><<<grid, block, shared_mem_size, stream>>>(
                output, input, N, C, H, W, total_spatial);
            break;
        case 128:
            log_softmax_dim1_kernel<128><<<grid, block, shared_mem_size, stream>>>(
                output, input, N, C, H, W, total_spatial);
            break;
        case 256:
            log_softmax_dim1_kernel<256><<<grid, block, shared_mem_size, stream>>>(
                output, input, N, C, H, W, total_spatial);
            break;
        default:
            log_softmax_dim1_kernel<256><<<grid, block, shared_mem_size, stream>>>(
                output, input, N, C, H, W, total_spatial);
            break;
    }
}

extern "C" void softmax_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int dim,
    cudaStream_t stream)
{
    if (dim != 1) return;
    int total_spatial = N * H * W;
    int block_size = 32;
    while (block_size < C) block_size <<= 1;
    if (block_size > 256) block_size = 256;

    int shared_mem_size = 2 * block_size * sizeof(float);

    dim3 grid(total_spatial);
    dim3 block(block_size);

    switch(block_size) {
        case 32:
            softmax_dim1_kernel<32><<<grid, block, shared_mem_size, stream>>>(
                output, input, N, C, H, W, total_spatial);
            break;
        case 64:
            softmax_dim1_kernel<64><<<grid, block, shared_mem_size, stream>>>(
                output, input, N, C, H, W, total_spatial);
            break;
        case 128:
            softmax_dim1_kernel<128><<<grid, block, shared_mem_size, stream>>>(
                output, input, N, C, H, W, total_spatial);
            break;
        case 256:
            softmax_dim1_kernel<256><<<grid, block, shared_mem_size, stream>>>(
                output, input, N, C, H, W, total_spatial);
            break;
        default:
            softmax_dim1_kernel<256><<<grid, block, shared_mem_size, stream>>>(
                output, input, N, C, H, W, total_spatial);
            break;
    }
}

extern "C" void reflection_pad2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int pad,
    cudaStream_t stream)
{
    int H_out = H_in + 2 * pad;
    int W_out = W_in + 2 * pad;
    int total = N * C * H_out * W_out;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    reflection_pad2d_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, N, C, H_in, W_in, H_out, W_out, pad, pad);
}
