#include <cuda_runtime.h>
#include <math.h>

// LayerNorm kernel: For each batch element n, normalize over C*H*W elements.
// We assign one block per batch element. Within block, use parallel reduction to compute sum and sum_sq.
// Then each thread normalizes its assigned elements using the computed statistics.
// Since normalized_shape is the trailing dimensions, and we have 4D, we normalise over dims 1,2,3.
// We assume the tensor may be non-contiguous; we pass strides for the original tensor (after abs).
// mean and rstd are computed using float accumulations to match PyTorch reasonably.

__global__ void layer_norm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    long long in_s0, long long in_s1, long long in_s2, long long in_s3,
    long long out_s0, long long out_s1, long long out_s2, long long out_s3,
    float eps)
{
    extern __shared__ float smem[]; // size 2 * blockDim.x for sum and sum_sq partials
    float* sum_partials = smem;               // size blockDim.x
    float* sum_sq_partials = &smem[blockDim.x]; // size blockDim.x

    int n = blockIdx.x; // batch index, assumes gridDim.x = N
    if (n >= N) return;

    int tid = threadIdx.x;
    int total = C * H * W; // number of elements to reduce per batch

    // Step 1: compute sum and sum_sq over this batch's elements using strided loop over total
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        // decompose idx to c,h,w within this batch
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp;
        long long offset = n * in_s0 + c * in_s1 + h * in_s2 + w * in_s3;
        float val = input[offset];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Store partials in shared memory
    sum_partials[tid] = local_sum;
    sum_sq_partials[tid] = local_sum_sq;
    __syncthreads();

    // Parallel reduction (tree) to get total sum and sum_sq
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_partials[tid] += sum_partials[tid + stride];
            sum_sq_partials[tid] += sum_sq_partials[tid + stride];
        }
        __syncthreads();
    }

    float mean = sum_partials[0] / total;
    float var = sum_sq_partials[0] / total - mean * mean;
    // Clamp variance to avoid negative due to numerical issues, matching PyTorch behavior
    if (var < 0.0f) var = 0.0f;
    float rstd = 1.0f / sqrtf(var + eps);

    // Step 2: normalize elements, same strided loop
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp;
        long long in_offset = n * in_s0 + c * in_s1 + h * in_s2 + w * in_s3;
        long long out_offset = n * out_s0 + c * out_s1 + h * out_s2 + w * out_s3;
        output[out_offset] = (input[in_offset] - mean) * rstd;
    }
}

// Launcher for layer_norm_forward
// output: output tensor data pointer (float)
// input: input tensor data pointer (float)
// dims: array of 4 ints [N,C,H,W]
// strides_in: array of 4 long long [s0,s1,s2,s3]
// strides_out: array of 4 long long [s0,s1,s2,s3]
// eps: epsilon
// stream: CUDA stream
extern "C" void layer_norm_forward_launcher(
    float* output,
    const float* input,
    const int* dims,
    const long long* strides_in,
    const long long* strides_out,
    float eps,
    cudaStream_t stream)
{
    int N = dims[0], C = dims[1], H = dims[2], W = dims[3];
    if (N == 0 || C == 0 || H == 0 || W == 0) return;
    // One block per batch element
    int block = 256; // reasonable
    // Shared memory size: 2 * block * sizeof(float)
    size_t smem = 2 * block * sizeof(float);
    layer_norm_kernel<<<N, block, smem, stream>>>(
        output, input, N, C, H, W,
        strides_in[0], strides_in[1], strides_in[2], strides_in[3],
        strides_out[0], strides_out[1], strides_out[2], strides_out[3],
        eps);
}

// Absolute value kernel with grid-stride loop, handling strides.
__global__ void abs_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    long long in_s0, long long in_s1, long long in_s2, long long in_s3,
    long long out_s0, long long out_s1, long long out_s2, long long out_s3)
{
    long long total = (long long)N * C * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total; i += stride) {
        // decompose i
        int w = i % W;
        long long tmp = i / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;
        long long in_offset = (long long)n * in_s0 + c * in_s1 + h * in_s2 + w * in_s3;
        long long out_offset = (long long)n * out_s0 + c * out_s1 + h * out_s2 + w * out_s3;
        output[out_offset] = fabsf(input[in_offset]);
    }
}

// Launcher for abs_forward
// output: output tensor data pointer (float)
// input: input tensor data pointer (float)
// dims: array of 4 ints [N,C,H,W]
// strides_in: array of 4 long long [s0,s1,s2,s3]
// strides_out: array of 4 long long [s0,s1,s2,s3]
// stream: CUDA stream
extern "C" void abs_forward_launcher(
    float* output,
    const float* input,
    const int* dims,
    const long long* strides_in,
    const long long* strides_out,
    cudaStream_t stream)
{
    int N = dims[0], C = dims[1], H = dims[2], W = dims[3];
    long long total = (long long)N * C * H * W;
    if (total == 0) return;
    int block = 256;
    // grid size: min(ceil(total/block), some cap)
    int grid = (int)((total + block - 1) / block);
    if (grid > 65535) grid = 65535; // max grid dim x
    abs_kernel<<<grid, block, 0, stream>>>(
        output, input, N, C, H, W,
        strides_in[0], strides_in[1], strides_in[2], strides_in[3],
        strides_out[0], strides_out[1], strides_out[2], strides_out[3]);
}
