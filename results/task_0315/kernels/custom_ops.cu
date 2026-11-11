#include <cuda_runtime.h>
#include <cmath>

// Fused tanh + LeakyReLU (negative_slope = 0.01) kernel.
// Input may have arbitrary strides; output is assumed to have the same shape as input,
// and we write in row-major order based on linear index.
// shape and stride arrays are passed by value (4 elements each).
__global__ void fused_tanh_leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t shape0, int64_t shape1, int64_t shape2, int64_t shape3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int ndim)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        int64_t rem = linear;
        int64_t in_offset = 0;
        // Compute input offset from linear index and input strides.
        // Output linear index assumes row-major order with shape.
        // Since input is 4D, we unroll the dimension loop.
        int64_t coord3 = rem % shape3; rem /= shape3;
        int64_t coord2 = rem % shape2; rem /= shape2;
        int64_t coord1 = rem % shape1; rem /= shape1;
        int64_t coord0 = rem; // rem should now be < shape0
        in_offset = coord0 * stride0 + coord1 * stride1 + coord2 * stride2 + coord3 * stride3;
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

    // We assume ndim == 4 for this task. Pass the arrays by value.
    // For safety, we always pass 4 elements; if ndim < 4, pad with 1s and 0 strides,
    // but the contract fixes shape to [512,32,128,128] so ndim=4.
    int64_t s0 = shape[0], s1 = shape[1], s2 = shape[2], s3 = shape[3];
    int64_t st0 = input_strides[0], st1 = input_strides[1], st2 = input_strides[2], st3 = input_strides[3];

    fused_tanh_leaky_relu_kernel<<<grid, block, 0, stream>>>(
        input, output, total_elements,
        s0, s1, s2, s3,
        st0, st1, st2, st3,
        ndim);
}
