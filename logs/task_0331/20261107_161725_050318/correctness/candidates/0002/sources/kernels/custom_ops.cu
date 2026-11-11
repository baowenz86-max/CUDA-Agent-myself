#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      return;                                                                   \
    }                                                                           \
  } while (0)

// GELU kernel: output = 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = idx; i < numel; i += stride) {
    float x = input[i];
    output[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
  }
}

// Greater-than kernel: output[i] = input[i] > threshold
__global__ void gt_kernel(const float* __restrict__ input,
                          bool* __restrict__ output,
                          float threshold,
                          int64_t numel) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = idx; i < numel; i += stride) {
    output[i] = (input[i] > threshold);
  }
}

// Broadcast addition kernel:
// input: bool tensor shape [batch_size, in_features]
// other: float tensor shape [in_features, 1, 1]
// output: float tensor shape [in_features, batch_size, in_features]
// Strides: input_stride0 = in_features, input_stride1 = 1
//          output_stride0 = batch_size * in_features, output_stride1 = in_features, output_stride2 = 1
__global__ void add_broadcast_kernel(const bool* __restrict__ input,
                                     const float* __restrict__ other,
                                     float* __restrict__ output,
                                     int64_t batch_size,
                                     int64_t in_features) {
  int64_t out_numel = in_features * batch_size * in_features;
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = idx; i < out_numel; i += stride) {
    // Decode output index (i, j, k) = (dim0, dim1, dim2)
    int64_t k = i % in_features;          // dim2
    int64_t rem = i / in_features;
    int64_t j = rem % batch_size;         // dim1
    int64_t d0 = rem / batch_size;        // dim0
    // input index: (j, k)
    bool input_val = input[j * in_features + k];
    // other index: (d0, 0, 0)
    float other_val = other[d0];
    output[i] = (input_val ? 1.0f : 0.0f) + other_val;
  }
}

// Custom linear kernel using tiled GEMM, avoiding cuBLAS layout issues.
// Tile sizes: BLOCK_SIZE = 16 (16x16 tile), THREADS = 256
constexpr int TILE_SIZE = 16;
constexpr int THREADS_PER_BLOCK = 256;

__global__ void linear_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int64_t batch_size,
                              int64_t in_features,
                              int64_t out_features) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int bx = blockIdx.x; // column tile index in output (out_features direction)
  int by = blockIdx.y; // row tile index in output (batch_size direction)
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float sum = 0.0f;
  int num_tiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; ++t) {
    // Load A tile: input[by*TILE_SIZE + ty][t*TILE_SIZE + tx]
    int a_row = by * TILE_SIZE + ty;
    int a_col = t * TILE_SIZE + tx;
    if (a_row < batch_size && a_col < in_features) {
      As[ty][tx] = input[a_row * in_features + a_col];
    } else {
      As[ty][tx] = 0.0f;
    }

    // Load B tile: weight[(bx*TILE_SIZE + ty)][t*TILE_SIZE + tx]
    int b_row = bx * TILE_SIZE + ty;
    int b_col = t * TILE_SIZE + tx;
    if (b_row < out_features && b_col < in_features) {
      Bs[ty][tx] = weight[b_row * in_features + b_col];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += As[ty][k] * Bs[tx][k];
    }

    __syncthreads();
  }

  int out_row = by * TILE_SIZE + ty;
  int out_col = bx * TILE_SIZE + tx;
  if (out_row < batch_size && out_col < out_features) {
    float val = sum;
    if (bias != nullptr) {
      val += bias[out_col];
    }
    output[out_row * out_features + out_col] = val;
  }
}

// Launchers with extern "C" linkage for the binding

extern "C" void linear_forward_launcher(const float* input,
                                        const float* weight,
                                        const float* bias,
                                        float* output,
                                        int64_t batch_size,
                                        int64_t in_features,
                                        int64_t out_features,
                                        cudaStream_t stream) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((out_features + TILE_SIZE - 1) / TILE_SIZE,
            (batch_size + TILE_SIZE - 1) / TILE_SIZE);
  linear_kernel<<<grid, block, 0, stream>>>(input, weight, bias, output,
                                             batch_size, in_features, out_features);
}

extern "C" void gelu_forward_launcher(const float* input,
                                       float* output,
                                       int64_t numel,
                                       cudaStream_t stream) {
  int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535; // grid cap, grid-stride loop handles rest
  gelu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void gt_forward_launcher(const float* input,
                                    bool* output,
                                    float threshold,
                                    int64_t numel,
                                    cudaStream_t stream) {
  int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;
  gt_kernel<<<blocks, threads, 0, stream>>>(input, output, threshold, numel);
}

extern "C" void add_forward_launcher(const bool* input,
                                     const float* other,
                                     float* output,
                                     int64_t batch_size,
                                     int64_t in_features,
                                     cudaStream_t stream) {
  int64_t out_numel = in_features * batch_size * in_features;
  int threads = 256;
  int blocks = (out_numel + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;
  add_broadcast_kernel<<<blocks, threads, 0, stream>>>(input, other, output,
                                                       batch_size, in_features);
}
