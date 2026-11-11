#include <cuda_runtime.h>
#include <cstdint>

// Element-wise Hardtanh (clamp) kernel supporting arbitrary strides.
// Each thread processes one element per iteration using a grid-stride loop.
template <int NDIM>
__global__ void hardtanh_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ strides,
    float min_val,
    float max_val)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        // Convert linear index to multidimensional coordinate using strides.
        int64_t offset = 0;
        int64_t rem = i;
        #pragma unroll
        for (int d = NDIM - 1; d >= 0; --d) {
            int64_t coord = rem % shape[d];
            rem /= shape[d];
            offset += coord * strides[d];
        }
        float val = input[offset];
        val = fminf(fmaxf(val, min_val), max_val);
        output[offset] = val;
    }
}

// Launcher for arbitrary number of dimensions (1 to 4 supported here; extend if needed).
// The binding layer should pass the number of dimensions and arrays for shape and strides.
extern "C" void hardtanh_launcher(
    float* output,
    const float* input,
    int64_t numel,
    int ndim,
    const int64_t* shape,
    const int64_t* strides,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    // Choose block size: 256 threads per block.
    const int block_size = 256;
    // Calculate grid size: enough blocks to cover all elements, but cap to avoid
    // exceeding grid dimension limits. Use grid-stride loop.
    int64_t blocks64 = (numel + block_size - 1) / block_size;
    // Limit grid to a reasonable maximum (e.g., 65535 blocks, which is max for x-dim in grid).
    // For large numel, we may need multiple iterations per thread.
    const int64_t max_blocks = 65535; // safe for all CUDA architectures
    int grid_size = (int)min(blocks64, max_blocks);
    
    // Dispatch based on ndim (up to 4, as input is 4D). For other cases, fallback to a generic
    // kernel that assumes contiguous (ndim not supported). Better: we can create specialized kernels.
    // Here we launch separate instantiations.
    switch (ndim) {
        case 1:
            hardtanh_kernel<1><<<grid_size, block_size, 0, stream>>>(
                output, input, numel, shape, strides, min_val, max_val);
            break;
        case 2:
            hardtanh_kernel<2><<<grid_size, block_size, 0, stream>>>(
                output, input, numel, shape, strides, min_val, max_val);
            break;
        case 3:
            hardtanh_kernel<3><<<grid_size, block_size, 0, stream>>>(
                output, input, numel, shape, strides, min_val, max_val);
            break;
        case 4:
            hardtanh_kernel<4><<<grid_size, block_size, 0, stream>>>(
                output, input, numel, shape, strides, min_val, max_val);
            break;
        default:
            // Unsupported dimension count; do nothing. Binding layer should check ndim <= 4.
            break;
    }
}
