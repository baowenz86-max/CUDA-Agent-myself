#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Utility macro and error checking
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
              cudaGetErrorString(err));                                      \
      return;                                                                \
    }                                                                        \
  } while (0)

// -----------------------------------------------------------------------------
// Kernel: copy tensor data (same shape, arbitrary strides) [squeeze_copy]
// -----------------------------------------------------------------------------
__global__ void copy_tensor_kernel(const float* __restrict__ src,
                                   float* __restrict__ dst,
                                   int64_t total_elements) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    dst[idx] = src[idx];
  }
}

// -----------------------------------------------------------------------------
// Kernel: bilinear upsample 2x, align_corners=False
// Input:  input (N, C, H, W)
// Output: output (N, C, 2H, 2W)
// Sampling formula (align_corners=False):
//   src_x = (dst_x + 0.5) * (W / out_W) - 0.5
//   src_y = (dst_y + 0.5) * (H / out_H) - 0.5
// Clamp to [0, W-1] and [0, H-1].
// -----------------------------------------------------------------------------
__global__ void upsample_bilinear_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int N, int C, int H, int W) {
  int out_H = H * 2;
  int out_W = W * 2;
  int64_t total_pixels = (int64_t)N * C * out_H * out_W;
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (idx >= total_pixels) return;

  int w_out = idx % out_W;
  int64_t tmp = idx / out_W;
  int h_out = tmp % out_H;
  tmp = tmp / out_H;
  int c = tmp % C;
  int n = tmp / C;

  // Compute source coordinates using half-pixel centers
  float src_x = (static_cast<float>(w_out) + 0.5f) * (static_cast<float>(W) / static_cast<float>(out_W)) - 0.5f;
  float src_y = (static_cast<float>(h_out) + 0.5f) * (static_cast<float>(H) / static_cast<float>(out_H)) - 0.5f;

  // Clamp to valid range
  if (src_x < 0.0f) src_x = 0.0f;
  if (src_y < 0.0f) src_y = 0.0f;
  if (src_x > static_cast<float>(W - 1)) src_x = static_cast<float>(W - 1);
  if (src_y > static_cast<float>(H - 1)) src_y = static_cast<float>(H - 1);

  int x0 = static_cast<int>(src_x);
  int y0 = static_cast<int>(src_y);
  int x1 = (x0 < W - 1) ? (x0 + 1) : x0;
  int y1 = (y0 < H - 1) ? (y0 + 1) : y0;
  float fx = src_x - static_cast<float>(x0);
  float fy = src_y - static_cast<float>(y0);

  const float* base = input + ((int64_t)n * C + c) * H * W;
  float val00 = base[y0 * W + x0];
  float val01 = base[y0 * W + x1];
  float val10 = base[y1 * W + x0];
  float val11 = base[y1 * W + x1];

  float top = val00 * (1.0f - fx) + val01 * fx;
  float bottom = val10 * (1.0f - fx) + val11 * fx;
  float result = top * (1.0f - fy) + bottom * fy;

  output[idx] = result;
}

// -----------------------------------------------------------------------------
// Launcher: squeeze_copy
// Input/output are both contiguous NCHW float32 tensors.
// -----------------------------------------------------------------------------
extern "C" void squeeze_copy_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream) {
  const int block_size = 256;
  int64_t blocks = (numel + block_size - 1) / block_size;
  if (blocks > 0) {
    copy_tensor_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        input, output, numel);
    CUDA_CHECK(cudaGetLastError());
  }
}

// -----------------------------------------------------------------------------
// Launcher: upsample_bilinear2d (custom kernel)
// Input tensor format: NCHW float32, dimensions [N, C, H, W]
// Output: [N, C, H*2, W*2] (scale_factor=2.0)
// align_corners = false, PyTorch default.
// -----------------------------------------------------------------------------
extern "C" void upsample_bilinear2d_launcher(const float* input,
                                             float* output,
                                             int N, int C, int H, int W,
                                             cudaStream_t stream) {
  int out_H = H * 2;
  int out_W = W * 2;
  int64_t total_elements = (int64_t)N * C * out_H * out_W;
  const int block_size = 256;
  int64_t blocks = (total_elements + block_size - 1) / block_size;
  if (blocks > 0) {
    upsample_bilinear_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        input, output, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());
  }
}
