#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

// BatchNorm inference kernel: y = (x - mean) / sqrt(var + eps) * weight + bias
// Input x: shape [N, C, H, W] with arbitrary strides (but typically contiguous)
// Output: same shape and strides as input, but we will allocate contiguous in binding, so can ignore strides for output if we write to same linear index based on input strides.
// However, binding may allocate contiguous output and pass output strides. For correctness, launcher will accept output strides as well and compute output offset with output strides.
__global__ void batch_norm_inference_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* __restrict__ x_strides, // [4]
    const int64_t* __restrict__ y_strides  // [4]
) {
    int64_t total = N * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        // recover n,c,h,w from linear idx assuming row-major over (N,C,H,W) in index space
        int64_t tmp = idx;
        int64_t w = tmp % W;
        tmp /= W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        int64_t x_idx = n * x_strides[0] + c * x_strides[1] + h * x_strides[2] + w * x_strides[3];
        int64_t y_idx = n * y_strides[0] + c * y_strides[1] + h * y_strides[2] + w * y_strides[3];

        float val = x[x_idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float wgt = weight[c];
        float b = bias[c];
        float inv_std = rsqrtf(var + eps);
        y[y_idx] = (val - mean) * inv_std * wgt + b;
    }
}

// Max reduce along dimension 1 (channels) for float32.
// Input x: shape [N, C, H, W] with strides x_strides[4].
// Output: shape [N, H, W] contiguous, effectively output_strides = [H*W, W, 1].
// We use one block per output element (n,h,w), with threads cooperating over C.
// C may be up to a few hundred, so blockDim <= 1024.
__global__ void max_reduce_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* __restrict__ x_strides // [4]
) {
    // One block per output element: n, h, w
    int64_t out_idx = blockIdx.x; // linear index over N*H*W
    if (out_idx >= N * H * W) return;

    int64_t n = out_idx / (H * W);
    int64_t rem = out_idx - n * H * W;
    int64_t h = rem / W;
    int64_t w = rem % W;

    // Base offset for this n,h,w at channel 0
    int64_t base = n * x_strides[0] + h * x_strides[2] + w * x_strides[3];

    float max_val = -INFINITY;
    // Not using shared memory since C is small and simple loop is fine. But shared memory reduction is possible.
    // We'll do a strided loop that each thread handles a subset of channels, then warp shuffle reduction and then block reduction.
    // Since C can be e.g. 64, and blockDim.x chosen to be e.g. 256, many threads idle. More efficient: one warp per output element, but that would require multiple output elements per block.
    // Alternative: each thread computes max over a subset of C, then reduce across block using shared memory.
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Loop over channels with stride block_size, accumulate local max
    float local_max = -INFINITY;
    for (int64_t c = tid; c < C; c += block_size) {
        float val = x[base + c * x_strides[1]];
        local_max = fmaxf(local_max, val);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[out_idx] = sdata[0];
    }
}

// Launcher for batch_norm_inference. The binding will allocate the output tensor with the same shape and strides as input (or contiguous if requested).
// We pass both x and y strides.
extern "C" void batch_norm_inference_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* x_strides,
    const int64_t* y_strides,
    cudaStream_t stream
) {
    int64_t total = N * C * H * W;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // Grid-stride loop so no need to cap exactly, but cap to a reasonable number to avoid too many blocks? Cannot exceed int max, but total ~134M / 256 ~ 524288 blocks, which is huge but may be okay? Actually maximum grid x dimension is 2^31-1, so 524288 is fine.
    batch_norm_inference_kernel<<<blocks, threads, 0, stream>>>(
        x, y, weight, bias, running_mean, running_var, eps,
        N, C, H, W, x_strides, y_strides);
}

// Launcher for max_reduce. output is contiguous shape [N,H,W].
extern "C" void max_reduce_launcher(
    const float* x,
    float* out,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* x_strides,
    cudaStream_t stream
) {
    int64_t out_numel = N * H * W;
    // block size: choose based on C. Typically C=64, use 64 or 128? Use 256 threads to handle future larger C, but for C < block size many threads idle. We'll choose block size as min(next_power_of_two(C), 256) maybe.
    int block_size = 256;
    while (block_size > 1 && block_size / 2 >= C) {
        block_size /= 2;
    }
    if (block_size > 1024) block_size = 1024;
    int blocks = (int)out_numel; // one block per output element, may be up to 16*1024*128 = 2,097,152 blocks, okay but grid.x max 2^31-1 so fine.
    // shared memory size: block_size floats
    size_t shared_mem = block_size * sizeof(float);
    max_reduce_dim1_kernel<<<blocks, block_size, shared_mem, stream>>>(
        x, out, N, C, H, W, x_strides);
}
