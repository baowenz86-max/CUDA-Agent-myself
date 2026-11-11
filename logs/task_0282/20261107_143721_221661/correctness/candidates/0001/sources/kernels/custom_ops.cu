// kernels/custom_ops.cu
// CUDA implementation for instance_norm_forward
// No torch/ATen/c10/pybind dependencies
#include <cuda_runtime.h>

// Helper to convert linear index to 5D coordinates using given strides
__device__ inline void index_to_nd(int64_t idx, int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
                                   int64_t* n, int64_t* c, int64_t* d, int64_t* h, int64_t* w) {
    *w = idx % W;
    idx /= W;
    *h = idx % H;
    idx /= H;
    *d = idx % D;
    idx /= D;
    *c = idx % C;
    *n = idx / C;
}

// Kernel to initialize mean and var arrays to 0.0
__global__ void init_reduction_buffers_kernel(double* mean, double* var, int64_t num_channels_total) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; i < num_channels_total; i += stride) {
        mean[i] = 0.0;
        var[i] = 0.0;
    }
}

// Kernel to compute per (N,C) sum and sum of squares using grid-stride loop.
// Input has arbitrary strides, so we use stride arrays.
__global__ void compute_mean_var_kernel(const float* __restrict__ input,
                                        double* __restrict__ mean,
                                        double* __restrict__ var,
                                        int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
                                        const int64_t* __restrict__ stride,
                                        int64_t total_elems) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t gstride = blockDim.x * gridDim.x;
    int64_t spatial_size = D * H * W;

    for (int64_t idx = tid; idx < total_elems; idx += gstride) {
        int64_t n, c, d, h, w;
        index_to_nd(idx, N, C, D, H, W, &n, &c, &d, &h, &w);
        int64_t input_offset = n * stride[0] + c * stride[1] + d * stride[2] + h * stride[3] + w * stride[4];
        float val = input[input_offset];
        // Use atomicAdd on double; may be slower but correct for arbitrary strides and no order dependence.
        // Since we are in compute capability >= 6.0, double atomicAdd is supported.
        int64_t channel_index = n * C + c;
        double val_d = (double)val;
        atomicAdd(&mean[channel_index], val_d);
        atomicAdd(&var[channel_index], val_d * val_d);
    }
}

// Normalize kernel: y = (x - mean) / sqrt(var / N + eps)
__global__ void normalize_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 const double* __restrict__ mean,
                                 const double* __restrict__ var,
                                 int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
                                 const int64_t* __restrict__ stride,
                                 int64_t total_elems,
                                 float eps) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t gstride = blockDim.x * gridDim.x;
    int64_t spatial_size = D * H * W;

    for (int64_t idx = tid; idx < total_elems; idx += gstride) {
        int64_t n, c, d, h, w;
        index_to_nd(idx, N, C, D, H, W, &n, &c, &d, &h, &w);
        int64_t input_offset = n * stride[0] + c * stride[1] + d * stride[2] + h * stride[3] + w * stride[4];
        int64_t channel_index = n * C + c;
        double mean_val = mean[channel_index] / (double)spatial_size;
        double var_val = var[channel_index] / (double)spatial_size - mean_val * mean_val;
        // Clamp variance to non-negative to avoid numerical issues
        if (var_val < 0.0) var_val = 0.0;
        float inv_std = rsqrtf((float)(var_val + (double)eps));
        float x_val = input[input_offset];
        output[input_offset] = (x_val - (float)mean_val) * inv_std;
    }
}

// Extern "C" launcher for the binding layer.
// The binding layer is responsible for allocating output, mean, var, stride array,
// copying strides to device, and getting the current CUDA stream.
// Here we accept raw pointers and dimensions.
extern "C" void instance_norm_forward_launcher(
    const float* input,
    float* output,
    double* mean,
    double* var,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const int64_t* stride,
    int64_t total_elems,
    float eps,
    cudaStream_t stream) {

    if (total_elems == 0 || N == 0 || C == 0 || D == 0 || H == 0 || W == 0) {
        return;
    }

    int64_t num_channels_total = N * C;

    // Initialize reduction buffers to zero
    int init_block = 256;
    int init_grid = (int)((num_channels_total + init_block - 1) / init_block);
    if (init_grid > 0) {
        init_reduction_buffers_kernel<<<init_grid, init_block, 0, stream>>>(mean, var, num_channels_total);
    }

    // Launch reduction kernel
    int block = 256;
    // Use a grid that covers the data with a reasonable number of blocks.
    // Grid-stride loop ensures all elements are processed.
    int max_blocks_per_launch = 65535; // safe for all current GPUs, but we can calculate
    int needed_blocks = (int)((total_elems + block - 1) / block);
    int grid = needed_blocks < max_blocks_per_launch ? needed_blocks : max_blocks_per_launch;
    if (grid == 0) grid = 1;

    compute_mean_var_kernel<<<grid, block, 0, stream>>>(
        input, mean, var,
        N, C, D, H, W,
        stride,
        total_elems);

    // Launch normalize kernel
    normalize_kernel<<<grid, block, 0, stream>>>(
        input, output,
        mean, var,
        N, C, D, H, W,
        stride,
        total_elems,
        eps);

    // Note: Error checking is left to the caller after synchronizing, but we can
    // optionally call cudaGetLastError here to catch launch errors immediately.
    // Since we don't want to synchronize, we'll just check the last error.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a pure launcher, we cannot throw; we can only ignore or print.
        // Binding layer may handle. We'll do nothing for now.
    }
}
