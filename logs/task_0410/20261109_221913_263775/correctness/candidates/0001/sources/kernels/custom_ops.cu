#include <cuda_runtime.h>

// CUDA kernel for embedding lookup: output[b,t,f,:] = weight[input[b,t,f], :]
// input: int64_t indices with shape [B, T, F]
// weight: float embedding matrix [num_embeddings, embedding_dim]
// output: float tensor [B, T, F, embedding_dim]
// All tensors are assumed to be on CUDA device. Strides are provided for generality.
__global__ void embedding_forward_kernel(
    const int64_t* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int total_indices,
    int embedding_dim,
    long long input_stride_b,
    long long input_stride_t,
    long long input_stride_f,
    long long output_stride_b,
    long long output_stride_t,
    long long output_stride_f,
    long long output_stride_e,
    long long weight_stride_0,
    long long weight_stride_1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_indices; i += stride) {
        // Convert linear index i to (b, t, f)
        int f = i % (input_stride_b / input_stride_t / input_stride_f); // not generally correct if strides not uniform, but assuming contiguous-like layout; better calculate from provided strides
        // Actually we need to compute indices using strides. We know that input is [B,T,F] with strides input_stride_b, input_stride_t, input_stride_f.
        // But total_indices = B*T*F, and we can compute b,t,f from linear index using division/modulo based on sizes, not strides.
        // However, we don't have B,T,F sizes directly. We can infer them from total_indices and strides? We assume standard sizes B,T,F such that total_indices = B*T*F.
        // Since we don't have sizes, we can compute offsets using strides and the linear index using the following approach:
        // We assume that the strides are consistent with contiguous storage (i.e., input_stride_b = T*F, input_stride_t = F, input_stride_f = 1).
        // But to support arbitrary strides, we need to know dimensions. The launcher will pass the sizes as well. Let's change signature to include sizes.
    }
}

// Revised kernel with sizes passed
__global__ void embedding_forward_kernel_sized(
    const int64_t* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int B,
    int T,
    int F,
    int embedding_dim,
    long long input_stride_b,
    long long input_stride_t,
    long long input_stride_f,
    long long output_stride_b,
    long long output_stride_t,
    long long output_stride_f,
    long long output_stride_e,
    long long weight_stride_0,
    long long weight_stride_1
) {
    int total = B * T * F;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int b = i / (T * F);
        int rem = i % (T * F);
        int t = rem / F;
        int f = rem % F;
        int64_t index = input[b * input_stride_b + t * input_stride_t + f * input_stride_f];
        // Assume index is valid (0 <= index < num_embeddings)
        const float* src = weight + index * weight_stride_0;
        float* dst = output + b * output_stride_b + t * output_stride_t + f * output_stride_f;
        for (int e = 0; e < embedding_dim; ++e) {
            dst[e * output_stride_e] = src[e * weight_stride_1];
        }
    }
}

// Kernel for adaptive average pooling to (1,1): output[b,t,0,0] = mean over H,W
__global__ void adaptive_avg_pool2d_1x1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B,
    int T,
    int H,
    int W,
    long long input_stride_b,
    long long input_stride_t,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_b,
    long long output_stride_t
) {
    int total_bt = B * T;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int pool_size = H * W;
    for (int i = idx; i < total_bt; i += stride) {
        int b = i / T;
        int t = i % T;
        const float* input_base = input + b * input_stride_b + t * input_stride_t;
        float sum = 0.0f;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                sum += input_base[h * input_stride_h + w * input_stride_w];
            }
        }
        output[b * output_stride_b + t * output_stride_t] = sum / pool_size;
    }
}

// Launcher for embedding_forward
// This function will be called from binding.cpp with pointers and dimensions.
extern "C" void embedding_forward_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int B,
    int T,
    int F,
    int embedding_dim,
    long long input_stride_b,
    long long input_stride_t,
    long long input_stride_f,
    long long output_stride_b,
    long long output_stride_t,
    long long output_stride_f,
    long long output_stride_e,
    long long weight_stride_0,
    long long weight_stride_1,
    cudaStream_t stream
) {
    int total = B * T * F;
    // Choose block size and grid size; use grid-stride loop
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    // Cap grid size to avoid excessive blocks? We can use a reasonable cap, e.g., 65535, but for large B*T*F may exceed.
    // Better compute grid as min(needed, some max). But since we have grid-stride, any grid >=1 works but performance varies.
    // We'll use a moderate grid: min(grid_size, 1024) * block_size? Actually we can use a formula that ensures enough parallelism.
    int max_grid = 65535;
    int grid = grid_size;
    if (grid > max_grid) {
        grid = max_grid;
    }
    // If grid == 0 (total == 0), return early.
    if (total == 0) return;
    embedding_forward_kernel_sized<<<grid, block_size, 0, stream>>>(
        input, weight, output, B, T, F, embedding_dim,
        input_stride_b, input_stride_t, input_stride_f,
        output_stride_b, output_stride_t, output_stride_f, output_stride_e,
        weight_stride_0, weight_stride_1
    );
}

// Launcher for adaptive_avg_pool2d_1x1
extern "C" void adaptive_avg_pool2d_1x1_launcher(
    const float* input,
    float* output,
    int B,
    int T,
    int H,
    int W,
    long long input_stride_b,
    long long input_stride_t,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_b,
    long long output_stride_t,
    cudaStream_t stream
) {
    int total_bt = B * T;
    if (total_bt == 0) return;
    int block_size = 256;
    int grid_size = (total_bt + block_size - 1) / block_size;
    int max_grid = 65535;
    int grid = grid_size;
    if (grid > max_grid) grid = max_grid;
    adaptive_avg_pool2d_1x1_kernel<<<grid, block_size, 0, stream>>>(
        input, output, B, T, H, W,
        input_stride_b, input_stride_t, input_stride_h, input_stride_w,
        output_stride_b, output_stride_t
    );
}
