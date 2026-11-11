#include <cuda_runtime.h>
#include <math.h>

// ELU kernel: output = x if x > 0 else exp(x) - 1
__global__ void elu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim0, int dim1, int dim2, int dim3,
    int stride0, int stride1, int stride2, int stride3,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        int remaining = i;
        int d0 = remaining / (dim1 * dim2 * dim3);
        remaining -= d0 * (dim1 * dim2 * dim3);
        int d1 = remaining / (dim2 * dim3);
        remaining -= d1 * (dim2 * dim3);
        int d2 = remaining / dim3;
        int d3 = remaining - d2 * dim3;

        int offset = d0 * stride0 + d1 * stride1 + d2 * stride2 + d3 * stride3;
        float x = input[offset];
        float y = (x > 0.0f) ? x : (expf(x) - 1.0f);
        output[offset] = y;
    }
}

// Tanh kernel: output = tanh(x)
__global__ void tanh_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim0, int dim1, int dim2, int dim3,
    int stride0, int stride1, int stride2, int stride3,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        int remaining = i;
        int d0 = remaining / (dim1 * dim2 * dim3);
        remaining -= d0 * (dim1 * dim2 * dim3);
        int d1 = remaining / (dim2 * dim3);
        remaining -= d1 * (dim2 * dim3);
        int d2 = remaining / dim3;
        int d3 = remaining - d2 * dim3;

        int offset = d0 * stride0 + d1 * stride1 + d2 * stride2 + d3 * stride3;
        float x = input[offset];
        float y = tanhf(x);
        output[offset] = y;
    }
}

// Helper to compute grid size based on total elements and block size
inline int compute_grid_size(int total_elements, int block_size) {
    int grid = (total_elements + block_size - 1) / block_size;
    // Cap grid size to avoid excessive large grid for big tensors
    if (grid > 65535) {
        grid = 65535;
    }
    return grid;
}

// Host launchers for binding layer
extern "C" void elu_forward_launcher(
    const float* input,
    float* output,
    int dim0, int dim1, int dim2, int dim3,
    int stride0, int stride1, int stride2, int stride3,
    int total_elements,
    cudaStream_t stream)
{
    if (total_elements == 0) return;
    int block_size = 256;
    int grid_size = compute_grid_size(total_elements, block_size);
    elu_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output,
        dim0, dim1, dim2, dim3,
        stride0, stride1, stride2, stride3,
        total_elements);
}

extern "C" void tanh_forward_launcher(
    const float* input,
    float* output,
    int dim0, int dim1, int dim2, int dim3,
    int stride0, int stride1, int stride2, int stride3,
    int total_elements,
    cudaStream_t stream)
{
    if (total_elements == 0) return;
    int block_size = 256;
    int grid_size = compute_grid_size(total_elements, block_size);
    tanh_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output,
        dim0, dim1, dim2, dim3,
        stride0, stride1, stride2, stride3,
        total_elements);
}
