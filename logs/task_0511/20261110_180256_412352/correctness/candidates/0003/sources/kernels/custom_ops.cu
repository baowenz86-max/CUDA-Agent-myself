#include <cuda_runtime.h>

// Elementwise addition with broadcasting: out = x + parameter.
// x has shape [dim0, dim1, dim2, dim3] and strides [sx0, sx1, sx2, sx3].
// parameter has shape [dim0, dim1, 1, 1] (as per contract), contiguous, with strides [sp0, sp1].
// The output is written using the same strides as x.
// Grid-stride loop handles very large element counts.
__global__ void add_parameter_kernel(
    const float* __restrict__ x,
    const float* __restrict__ parameter,
    float* __restrict__ out,
    int64_t total,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t sx0, int64_t sx1, int64_t sx2, int64_t sx3,
    int64_t sp0, int64_t sp1)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        // Decompose linear index into 4D coordinates (assuming row-major layout as given by strides)
        int64_t i0 = i / (dim1 * dim2 * dim3);
        int64_t rem = i % (dim1 * dim2 * dim3);
        int64_t i1 = rem / (dim2 * dim3);
        rem = rem % (dim2 * dim3);
        int64_t i2 = rem / dim3;
        int64_t i3 = rem % dim3;

        // Compute linear offset into x using actual strides (supports non-contiguous)
        int64_t x_offset = i0 * sx0 + i1 * sx1 + i2 * sx2 + i3 * sx3;
        // Parameter index: broadcast from [dim0, dim1, 1, 1], so i2 and i3 are ignored.
        // parameter is contiguous with strides sp0, sp1.
        int64_t p_offset = i0 * sp0 + i1 * sp1;

        out[x_offset] = x[x_offset] + parameter[p_offset];
    }
}

// Launcher function called from binding.cpp
extern "C" void add_parameter_launcher(
    const float* x,
    const float* parameter,
    float* out,
    int64_t total,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t stride_x0,
    int64_t stride_x1,
    int64_t stride_x2,
    int64_t stride_x3,
    int64_t stride_param0,
    int64_t stride_param1,
    cudaStream_t stream)
{
    if (total <= 0) return;

    const int threads = 256;
    // Cap blocks to avoid excessive grid size; grid-stride loop handles the rest.
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // 1D grid max limit (actually can be larger, but this is safe)
    dim3 grid_dim(static_cast<unsigned int>(blocks));

    add_parameter_kernel<<<grid_dim, threads, 0, stream>>>(
        x, parameter, out, total,
        dim0, dim1, dim2, dim3,
        stride_x0, stride_x1, stride_x2, stride_x3,
        stride_param0, stride_param1);
}
