#include <cuda_runtime.h>

// Kernel for torch.max over a single dimension (dim_size small)
// Each thread computes one output element (b, h, w) by scanning dim_size values.
template <typename T>
__global__ void max_dim_kernel(const T* __restrict__ input,
                               T* __restrict__ values,
                               int64_t* __restrict__ indices,
                               int batch, int height, int width, int dim_size) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t out_stride = (int64_t)gridDim.x * blockDim.x;
    int64_t total = (int64_t)batch * height * width;

    for (; out_idx < total; out_idx += out_stride) {
        int b = out_idx / (height * width);
        int rem = out_idx % (height * width);
        int h = rem / width;
        int w = rem % width;

        // Base offset: input[b, 0, h, w]
        int64_t in_base = ((int64_t)b * dim_size * height + 0 * height + h) * width + w;
        // assuming contiguous layout: shape [batch, dim_size, height, width]
        // stride: dim_size*height*width, height*width, width, 1

        T best_val = input[in_base];
        int64_t best_idx = 0;
        for (int d = 1; d < dim_size; ++d) {
            int64_t in_offset = in_base + (int64_t)d * height * width;
            T val = input[in_offset];
            if (val > best_val) {
                best_val = val;
                best_idx = d;
            }
        }
        values[out_idx] = best_val;
        indices[out_idx] = best_idx;
    }
}

// Kernel for torch.argmin over a single dimension
template <typename T>
__global__ void argmin_dim_kernel(const T* __restrict__ input,
                                  int64_t* __restrict__ indices,
                                  int batch, int height, int width, int dim_size) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t out_stride = (int64_t)gridDim.x * blockDim.x;
    int64_t total = (int64_t)batch * height * width;

    for (; out_idx < total; out_idx += out_stride) {
        int b = out_idx / (height * width);
        int rem = out_idx % (height * width);
        int h = rem / width;
        int w = rem % width;

        int64_t in_base = ((int64_t)b * dim_size * height + 0 * height + h) * width + w;

        T best_val = input[in_base];
        int64_t best_idx = 0;
        for (int d = 1; d < dim_size; ++d) {
            int64_t in_offset = in_base + (int64_t)d * height * width;
            T val = input[in_offset];
            if (val < best_val) {
                best_val = val;
                best_idx = d;
            }
        }
        indices[out_idx] = best_idx;
    }
}

// Launcher for custom_max_dim
// input is assumed contiguous float32 tensor of shape [batch, dim_size, height, width]
// values output float32 [batch, height, width]
// indices output int64 [batch, height, width]
extern "C" void custom_max_dim_launcher(const float* input,
                                        float* values,
                                        int64_t* indices,
                                        int64_t batch,
                                        int64_t dim_size,
                                        int64_t height,
                                        int64_t width,
                                        cudaStream_t stream) {
    int64_t total = batch * height * width;
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    // cap grid size to avoid excessive blocks if total is small
    if (grid_size > 65535) grid_size = 65535;

    max_dim_kernel<float><<<(int)grid_size, block_size, 0, stream>>>(
        input, values, indices,
        (int)batch, (int)height, (int)width, (int)dim_size);
}

// Launcher for custom_argmin_dim
extern "C" void custom_argmin_dim_launcher(const float* input,
                                           int64_t* indices,
                                           int64_t batch,
                                           int64_t dim_size,
                                           int64_t height,
                                           int64_t width,
                                           cudaStream_t stream) {
    int64_t total = batch * height * width;
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    argmin_dim_kernel<float><<<(int)grid_size, block_size, 0, stream>>>(
        input, indices,
        (int)batch, (int)height, (int)width, (int)dim_size);
}
