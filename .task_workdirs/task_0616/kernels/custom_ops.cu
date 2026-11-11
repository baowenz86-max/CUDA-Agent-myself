#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// kernel: sum over channel dimension (dim=1) with keepdim
// Input: x [N, C, H, W] contiguous float32
// Output: out [N, 1, H, W] contiguous float32
// Each thread handles one (n, h, w) output element, summing over C channels.
// Grid-stride loop over total output elements N*H*W.
// -----------------------------------------------------------------------------
__global__ void sum_dim1_keepdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W) {
  int total = N * H * W;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < total; i += stride) {
    int w = i % W;
    int tmp = i / W;
    int h = tmp % H;
    int n = tmp / H;

    const float* base = x + ((long long)n * C + 0) * H * W + h * W + w;
    float sum = base[0];
    // unroll first 4 channels for better performance (C often small)
    int c = 1;
    for (; c + 3 < C; c += 4) {
      sum += base[c * H * W];
      sum += base[(c + 1) * H * W];
      sum += base[(c + 2) * H * W];
      sum += base[(c + 3) * H * W];
    }
    for (; c < C; ++c) {
      sum += base[c * H * W];
    }

    out[(long long)n * H * W + h * W + w] = sum;
  }
}

// -----------------------------------------------------------------------------
// kernel: elementwise sigmoid, stable for large |x|
// Input: x [N, 1, H, W] contiguous float32
// Output: out [same shape] contiguous float32
// -----------------------------------------------------------------------------
__global__ void sigmoid_elementwise_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < total; i += stride) {
    float v = x[i];
    // stable sigmoid: 1/(1+exp(-|v|)) and then handle sign
    float exp_neg_abs = __expf(-fabsf(v));
    float sig = 1.0f / (1.0f + exp_neg_abs);
    out[i] = (v >= 0.0f) ? sig : (1.0f - sig);
  }
}

// -----------------------------------------------------------------------------
// Host launcher for sum_dim1_keepdim
// Input pointer x points to contiguous float32 CUDA tensor of shape [N,C,H,W]
// Output pointer out points to allocated float32 CUDA tensor of shape [N,1,H,W]
// -----------------------------------------------------------------------------
extern "C" void sum_dim1_keepdim_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    cudaStream_t stream) {
  int total = N * H * W;
  if (total <= 0) {
    return;  // empty tensor, nothing to do
  }

  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  // Cap grid to avoid too many blocks; use grid-stride loop.
  if (blocks > 4096) blocks = 4096;

  sum_dim1_keepdim_kernel<<<blocks, threads, 0, stream>>>(
      x, out, N, C, H, W);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Cannot use exceptions here; launcher is C-compatible.
    // The caller (binding.cpp) should check cudaGetLastError after launch.
    // We leave error checking to the binding layer via cudaGetLastError.
  }
}

// -----------------------------------------------------------------------------
// Host launcher for sigmoid_elementwise
// Input pointer x points to contiguous float32 CUDA tensor of shape [N,1,H,W]
// Output pointer out points to allocated float32 CUDA tensor of same shape
// -----------------------------------------------------------------------------
extern "C" void sigmoid_elementwise_launcher(
    const float* x,
    float* out,
    int total,
    cudaStream_t stream) {
  if (total <= 0) {
    return;
  }

  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  if (blocks > 4096) blocks = 4096;

  sigmoid_elementwise_kernel<<<blocks, threads, 0, stream>>>(
      x, out, total);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Errors are reported by binding layer after launch.
  }
}
