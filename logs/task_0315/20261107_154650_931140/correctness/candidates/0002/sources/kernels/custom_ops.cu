#include <cuda_runtime.h>
#include <cmath>

// Fused tanh + LeakyReLU (negative_slope = 0.01) kernel.
// Input may have arbitrary strides; output is assumed to have the same shape as input,
// and we write in row-major order based on linear index.
// We are given input_strides (array of length ndim) and shape (array of length ndim)
// so that we can compute the input offset for each linear output index.
__global__ void fused_tanh_leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ shape,
    int ndim)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        int64_t rem = linear;
        int64_t in_offset = 0;
        // Compute input offset from linear index and input strides.
        // Output linear index assumes row-major order with shape.
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t coord = rem % shape[d];
            rem /= shape[d];
            in_offset += coord * input_strides[d];
        }
        float val = input[in_offset];
        float tanh_val = tanhf(val);
        float out_val = (tanh_val > 0.0f) ? tanh_val : 0.01f * tanh_val;
        output[linear] = out_val;
    }
}

// Launcher function called from binding.
extern "C" void fused_tanh_leaky_relu_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* shape,
    int ndim,
    cudaStream_t stream)
{
    const int block_size = 256;
    // Grid size capped at 2^31-1, but use a reasonable number based on total_elements.
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding max grid dimension.
    const int64_t max_grid_x = 2147483647; // 2^31 - 1
    if (grid_size > max_grid_x) {
        grid_size = max_grid_x;
    }
    dim3 grid(static_cast<unsigned int>(grid_size));
    dim3 block(block_size);

    fused_tanh_leaky_relu_kernel<<<grid, block, 0, stream>>>(
        input, output, total_elements, input_strides, shape, ndim);
}
