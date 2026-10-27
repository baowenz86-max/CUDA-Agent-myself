#include <cuda_runtime.h>

// Elementwise leaky ReLU kernel.
// x is a 3D tensor of shape [N, C, H, W] but we treat it as a flattened
template <typename scalar_t>
__global__ void leaky_relu_kernel_contiguous(
    scalar_t* out,
    const scalar_t* in,
    const int64_t numel,
    const scalar_t negative_slope) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (; idx < numel; idx += stride) {
    scalar_t v = in[idx];
    out[idx] = (v >= static_cast<scalar_t>(0)) ? v : negative_slope * v;
  }
}

// Generic kernel for non-contiguous input/output with same strides.
// We assume out has same strides as in, so index computation is identical.
template <typename scalar_t>
__global__ void leaky_relu_kernel_generic(
    scalar_t* out,
    const scalar_t* in,
    const int64_t numel,
    const int64_t dim0_size,
    const int64_t dim1_size,
    const int64_t dim2_size,
    const int64_t stride0,
    const int64_t stride1,
    const int64_t stride2,
    const scalar_t negative_slope) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (; idx < numel; idx += stride) {
    int64_t i = idx / (dim1_size * dim2_size);
    int64_t rem = idx % (dim1_size * dim2_size);
    int64_t j = rem / dim2_size;
    int64_t k = rem % dim2_size;
    int64_t offset = i * stride0 + j * stride1 + k * stride2;
    scalar_t v = in[offset];
    out[offset] = (v >= static_cast<scalar_t>(0)) ? v : negative_slope * v;
  }
}

extern "C" void leaky_relu_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    float negative_slope,
    cudaStream_t stream) {
  if (numel == 0) return;

  // If strides are contiguous-like (stride2==1, stride1==dim2, stride0==dim1*dim2)
  // use contiguous kernel for better performance.
  bool contiguous = (stride2 == 1 && stride1 == dim2 && stride0 == dim1 * dim2);

  const int64_t threads = 256;
  const int64_t blocks = min((numel + threads - 1) / threads, static_cast<int64_t>(65535));
  // If blocks would exceed grid max, grid-stride handles it anyway.
  // Choose grid size at most 65535 or based on occupancy.
  int grid = static_cast<int>(blocks > 0 ? blocks : 1);

  if (contiguous) {
    leaky_relu_kernel_contiguous<float><<<grid, threads, 0, stream>>>(
        output, input, numel, negative_slope);
  } else {
    leaky_relu_kernel_generic<float><<<grid, threads, 0, stream>>>(
        output, input, numel, dim0, dim1, dim2,
        stride0, stride1, stride2, negative_slope);
  }
}
