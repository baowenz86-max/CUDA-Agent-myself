#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for torch.triu(x, diagonal=1) on last two dimensions.
// Supports arbitrary batch dimensions and strides.
// x: input pointer, out: output pointer, total elements
// dim0: height, dim1: width, diagonal: as in torch.triu
template <typename scalar_t>
__global__ void triu_kernel(const scalar_t* __restrict__ x,
                            scalar_t* __restrict__ out,
                            int total_elements,
                            int dim0,
                            int dim1,
                            long long int stride0,
                            long long int stride1,
                            long long int batch_stride,
                            int diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        // Compute batch index, row, col
        int flat = i;
        int batch = flat / (dim0 * dim1);
        int rem = flat - batch * (dim0 * dim1);
        int row = rem / dim1;
        int col = rem - row * dim1;

        // Offset in memory
        long long int offset = batch * batch_stride + row * stride0 + col * stride1;

        // Condition: keep if col - row >= diagonal, else zero
        if (col - row < diagonal) {
            out[i] = scalar_t(0);
        } else {
            out[i] = x[offset];
        }
    }
}

// CUDA kernel for torch.abs(x)
template <typename scalar_t>
__global__ void abs_kernel(const scalar_t* __restrict__ x,
                           scalar_t* __restrict__ out,
                           int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        out[i] = fabsf(x[i]);
    }
}

// Launcher for custom_triu
extern "C" void custom_triu_launcher(const float* x,
                                     float* out,
                                     int total_elements,
                                     int dim0,
                                     int dim1,
                                     long long int stride0,
                                     long long int stride1,
                                     long long int batch_stride,
                                     int diagonal,
                                     cudaStream_t stream) {
    const int block = 256;
    int grid = (total_elements + block - 1) / block;
    if (grid > 65535) grid = 65535; // grid-stride loop handles rest
    triu_kernel<float><<<grid, block, 0, stream>>>(
        x, out, total_elements, dim0, dim1, stride0, stride1, batch_stride, diagonal);
}

// Launcher for custom_abs
extern "C" void custom_abs_launcher(const float* x,
                                    float* out,
                                    int total_elements,
                                    cudaStream_t stream) {
    const int block = 256;
    int grid = (total_elements + block - 1) / block;
    if (grid > 65535) grid = 65535;
    abs_kernel<float><<<grid, block, 0, stream>>>(x, out, total_elements);
}
