#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Kernel: hardsigmoid_forward
// y = clamp(x/6 + 0.5, 0, 1)
// ---------------------------------------------------------------------------
__global__ void hardsigmoid_kernel(const float *__restrict__ x,
                                   float *__restrict__ y, int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = idx; i < numel; i += stride) {
    float val = __fmaf_rn(x[i], 1.0f / 6.0f, 0.5f);
    y[i] = fminf(fmaxf(val, 0.0f), 1.0f);
  }
}

extern "C" void custom_hardsigmoid_forward_launcher(
    const float *x, float *y, int64_t numel, cudaStream_t stream) {
  if (numel == 0)
    return;
  int block_size = 256;
  int64_t grid_size = (numel + block_size - 1) / block_size;
  // Cap grid size to avoid exceeding max grid dimension but use grid-stride loop
  int max_grid = 65535;
  if (grid_size > max_grid)
    grid_size = max_grid;
  hardsigmoid_kernel<<<(int)grid_size, block_size, 0, stream>>>(x, y, numel);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: scale_bias_forward
// y = x * scale + bias
// ---------------------------------------------------------------------------
__global__ void scale_bias_kernel(const float *__restrict__ x,
                                  float *__restrict__ y, float scale,
                                  float bias, int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = idx; i < numel; i += stride) {
    y[i] = fmaf(x[i], scale, bias);
  }
}

extern "C" void custom_scale_bias_forward_launcher(
    const float *x, float *y, float scale, float bias, int64_t numel,
    cudaStream_t stream) {
  if (numel == 0)
    return;
  int block_size = 256;
  int64_t grid_size = (numel + block_size - 1) / block_size;
  int max_grid = 65535;
  if (grid_size > max_grid)
    grid_size = max_grid;
  scale_bias_kernel<<<(int)grid_size, block_size, 0, stream>>>(x, y, scale,
                                                                bias, numel);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: instance norm forward
// For each (n, c), compute mean and biased variance over H*W.
// Normalize: y = (x - mean) / sqrt(var + eps)
// Input is NCHW contiguous float32.
// We use a two-phase approach: first compute mean/var per (n,c), then normalize.
// ---------------------------------------------------------------------------

// Grid: one block per (n,c) or multiple blocks per (n,c) for reduction?
// Since H*W = 4096, one block of 256 threads can compute sum and sumsq with
// grid-stride loop. That is efficient enough. We'll use one block per channel,
// total blocks = N * C. N*C = 512*64 = 32768, which is less than maximum grid
// dimension in x (2^31-1) on modern GPUs but may exceed 65535 on some older,
// but we can set gridDim as dim3 with x up to 2^31 for CUDA 9+ but we should
// cap to 65535 and use blockIdx.x only? Simpler: use one block per channel with
// grid = N*C (no grid-stride across channels, each block handles one channel).
// N*C = 32768 which is fine for any GPU (max grid x >= 65535 for compute 3.0+,
// for older maybe 65535 but we assume modern).

__global__ void instance_norm_mean_var_kernel(
    const float *__restrict__ x, float *__restrict__ mean,
    float *__restrict__ var, int64_t N, int64_t C, int64_t HW, float eps) {
  int64_t nc = blockIdx.x; // index in [0, N*C-1]
  if (nc >= N * C)
    return;
  int64_t n = nc / C;
  int64_t c = nc - n * C;
  const float *channel = x + (n * C + c) * HW;

  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int64_t i = threadIdx.x; i < HW; i += blockDim.x) {
    float v = channel[i];
    sum += v;
    sumsq = fmaf(v, v, sumsq);
  }

  // block reduction
  __shared__ float s_sum[256];
  __shared__ float s_sumsq[256];
  s_sum[threadIdx.x] = sum;
  s_sumsq[threadIdx.x] = sumsq;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
      s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float mean_val = s_sum[0] / (float)HW;
    float var_val = fmaf(-mean_val, mean_val, s_sumsq[0] / (float)HW);
    // biased variance, ensure non-negative
    var_val = fmaxf(var_val, 0.0f);
    mean[nc] = mean_val;
    var[nc] = var_val;
  }
}

__global__ void instance_norm_normalize_kernel(
    const float *__restrict__ x, float *__restrict__ y,
    const float *__restrict__ mean, const float *__restrict__ var,
    int64_t N, int64_t C, int64_t HW, float eps) {
  int64_t nc = blockIdx.x; // channel index
  if (nc >= N * C)
    return;
  int64_t n = nc / C;
  int64_t c = nc - n * C;
  const float *channel_in = x + (n * C + c) * HW;
  float *channel_out = y + (n * C + c) * HW;
  float m = mean[nc];
  float v = var[nc];
  float inv_std = rsqrtf(v + eps);

  for (int64_t i = threadIdx.x; i < HW; i += blockDim.x) {
    channel_out[i] = (channel_in[i] - m) * inv_std;
  }
}

extern "C" void custom_instance_norm_forward_launcher(
    const float *x, float *y, float *mean, float *var, int64_t N, int64_t C,
    int64_t H, int64_t W, float eps, cudaStream_t stream) {
  if (N == 0 || C == 0 || H == 0 || W == 0)
    return;
  int64_t HW = H * W;
  int64_t total_channels = N * C;

  int block_size = 256;
  // grid = total_channels, ensure not exceeding max grid dimension
  // total_channels = 32768, fine on all modern GPUs; but if C=1,N huge?
  // we can cap and use grid-stride? For safety we use grid-stride on nc indexing.
  // But kernel currently uses blockIdx.x as nc without stride loop.
  // Modify kernel to use grid-stride loop over nc to be safe.
  // We'll rewrite kernels with grid-stride loop over channels.
  int grid_size = (int)min((int64_t)65535, total_channels);
  // We'll launch mean_var and normalize with that grid.
  instance_norm_mean_var_kernel<<<grid_size, block_size, 0, stream>>>(
      x, mean, var, N, C, HW, eps);
  CUDA_CHECK(cudaGetLastError());
  instance_norm_normalize_kernel<<<grid_size, block_size, 0, stream>>>(
      x, y, mean, var, N, C, HW, eps);
  CUDA_CHECK(cudaGetLastError());
}

// If total_channels > grid_size, the kernels above only process first grid_size
// channels because they return if nc >= N*C but they don't loop; they assume
// nc = blockIdx.x and then if nc>=N*C return, meaning channels>=grid_size are
// not processed. Need to add grid-stride loop over nc inside kernels.
// We'll fix by modifying kernels to use loop:
// for (int64_t nc = blockIdx.x; nc < N*C; nc += gridDim.x) { ... }
// And ensure mean/var arrays have N*C elements.

__global__ void instance_norm_mean_var_kernel_v2(
    const float *__restrict__ x, float *__restrict__ mean,
    float *__restrict__ var, int64_t N, int64_t C, int64_t HW, float eps) {
  for (int64_t nc = blockIdx.x; nc < N * C; nc += gridDim.x) {
    int64_t n = nc / C;
    int64_t c = nc - n * C;
    const float *channel = x + (n * C + c) * HW;
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < HW; i += blockDim.x) {
      float v = channel[i];
      sum += v;
      sumsq = fmaf(v, v, sumsq);
    }
    __shared__ float s_sum[256];
    __shared__ float s_sumsq[256];
    s_sum[threadIdx.x] = sum;
    s_sumsq[threadIdx.x] = sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + s];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float mean_val = s_sum[0] / (float)HW;
      float var_val = fmaf(-mean_val, mean_val, s_sumsq[0] / (float)HW);
      var_val = fmaxf(var_val, 0.0f);
      mean[nc] = mean_val;
      var[nc] = var_val;
    }
    // Important: after writing mean/var, we need to ensure all threads have
    // finished before the block moves to next channel, because shared memory
    // will be reused and we must not overwrite other channel's data? We don't
    // write mean/var except thread0, but need __syncthreads before next
    // iteration to reuse shared memory with different values.
    __syncthreads();
  }
}

__global__ void instance_norm_normalize_kernel_v2(
    const float *__restrict__ x, float *__restrict__ y,
    const float *__restrict__ mean, const float *__restrict__ var,
    int64_t N, int64_t C, int64_t HW, float eps) {
  for (int64_t nc = blockIdx.x; nc < N * C; nc += gridDim.x) {
    int64_t n = nc / C;
    int64_t c = nc - n * C;
    const float *channel_in = x + (n * C + c) * HW;
    float *channel_out = y + (n * C + c) * HW;
    float m = mean[nc];
    float v = var[nc];
    float inv_std = rsqrtf(v + eps);
    for (int64_t i = threadIdx.x; i < HW; i += blockDim.x) {
      channel_out[i] = (channel_in[i] - m) * inv_std;
    }
  }
}

// Final launcher uses v2 kernels
extern "C" void custom_instance_norm_forward_launcher(
    const float *x, float *y, float *mean, float *var, int64_t N, int64_t C,
    int64_t H, int64_t W, float eps, cudaStream_t stream) {
  if (N == 0 || C == 0 || H == 0 || W == 0)
    return;
  int64_t HW = H * W;
  int64_t total_channels = N * C;
  int block_size = 256;
  int max_grid = 65535; // safe cap
  int grid_size = (int)min((int64_t)max_grid, total_channels);
  instance_norm_mean_var_kernel_v2<<<grid_size, block_size, 0, stream>>>(
      x, mean, var, N, C, HW, eps);
  CUDA_CHECK(cudaGetLastError());
  instance_norm_normalize_kernel_v2<<<grid_size, block_size, 0, stream>>>(
      x, y, mean, var, N, C, HW, eps);
  CUDA_CHECK(cudaGetLastError());
}

// NOTE: We also need to define the original kernel names but they are not used.
// To avoid duplicate symbol confusion, we keep only v2 versions and the final launcher.
// Remove the earlier non-v2 kernels and launcher. We'll rewrite file accordingly.
