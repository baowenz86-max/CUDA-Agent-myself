// custom_ops.cu - CUDA kernels and launchers for linear_forward
// No torch/ATen/pybind includes; only raw CUDA and cuda_runtime.
// Binding code will be in a separate .cpp file.

#include <cuda_runtime.h>
#include <cstdint>

// Kernel configuration
// Block assumes: OUTPUT_TILE_M x OUTPUT_TILE_N threads per block for output tile
// We use 16x16 threads, each thread computes one output element.
// Shared memory: input tile (TILE_M x TILE_K) + weight tile (TILE_K x TILE_N)
// TILE_M = 16, TILE_N = 16, TILE_K = 16

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Kernel: linear_forward_kernel
// Computes output = input @ weight^T + bias for batched 3D input.
// input shape: [B, L, K], weight shape: [N, K], bias shape: [N]
// output shape: [B, L, N]
// Strides: input_stride_b, input_stride_l, input_stride_k (contiguous typically)
// weight is row-major [N,K] with strides weight_stride_n=N, weight_stride_k=1?
// Actually weight shape [N,K] means weight[j][k] = weight[j*K + k]
// We treat weight as row-major, can use weight_stride_n = K, weight_stride_k = 1 (or user provided).
// To handle non-contiguous weight, we pass strides.
// For simplicity, assume weight is contiguous row-major (K,1) but include strides in case.
__global__ void linear_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int L, int K, int N,
    int input_stride_b, int input_stride_l, int input_stride_k,
    int weight_stride_n, int weight_stride_k,
    int output_stride_b, int output_stride_l, int output_stride_n)
{
    // Block index: grid over output tiles (B*L/TILE_M x N/TILE_N)
    // Each block computes a TILE_M x TILE_N tile of output for a particular batch and seq chunk.
    int block_row = blockIdx.y; // index over B*L dimension, each block handles TILE_M rows
    int block_col = blockIdx.x; // index over N dimension, each block handles TILE_N cols

    // Shared memory for input tile and weight tile
    __shared__ float input_tile[TILE_M][TILE_K];
    __shared__ float weight_tile[TILE_K][TILE_N];

    // Determine output row start: row = block_row * TILE_M + threadIdx.y (within tile)
    int out_row_global = block_row * TILE_M + threadIdx.y; // -> b, l indices
    int out_col_global = block_col * TILE_N + threadIdx.x; // -> n index

    // Local accumulation
    float acc = 0.0f;

    // Loop over K dimension in tiles of TILE_K
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load input tile: input[b, l, k_tile + threadIdx.x] if within bounds
        // Input row is based on out_row_global (b*l). Need to map to b,l.
        int b = out_row_global / L;
        int l = out_row_global % L;
        // input index for element (b, l, k_tile + threadIdx.x)
        int k_idx = k_tile + threadIdx.x;
        if (out_row_global < B * L && k_idx < K) {
            input_tile[threadIdx.y][threadIdx.x] = input[b * input_stride_b + l * input_stride_l + k_idx * input_stride_k];
        } else {
            input_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load weight tile: weight[k_tile + threadIdx.y][out_col_global]
        // weight shape [N,K], so weight[n][k] = weight[n * weight_stride_n + k * weight_stride_k]
        int k_row = k_tile + threadIdx.y;
        if (out_col_global < N && k_row < K) {
            weight_tile[threadIdx.y][threadIdx.x] = weight[out_col_global * weight_stride_n + k_row * weight_stride_k];
        } else {
            weight_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Accumulate dot product for this tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += input_tile[threadIdx.y][kk] * weight_tile[kk][threadIdx.x];
        }

        __syncthreads();
    }

    // Add bias and write output
    if (out_row_global < B * L && out_col_global < N) {
        int b = out_row_global / L;
        int l = out_row_global % L;
        int output_idx = b * output_stride_b + l * output_stride_l + out_col_global * output_stride_n;
        output[output_idx] = acc + bias[out_col_global];
    }
}

// Launcher: linear_forward_launcher
// Extern "C" for binding
// Parameters:
//   input: pointer to input data (float) on device
//   weight: pointer to weight matrix (float) on device
//   bias: pointer to bias vector (float) on device
//   output: pointer to output data (float) on device
//   B, L, K, N: dimensions
//   input_stride_b, input_stride_l, input_stride_k: strides of input (elements)
//   weight_stride_n, weight_stride_k: strides of weight (elements)
//   output_stride_b, output_stride_l, output_stride_n: strides of output (elements)
//   stream: CUDA stream
// Note: All pointers are raw device pointers; binding layer handles checks and allocation.
extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t B, int64_t L, int64_t K, int64_t N,
    int64_t input_stride_b, int64_t input_stride_l, int64_t input_stride_k,
    int64_t weight_stride_n, int64_t weight_stride_k,
    int64_t output_stride_b, int64_t output_stride_l, int64_t output_stride_n,
    cudaStream_t stream)
{
    // Convert to int for kernel launch, check for grid size limits
    // B*L = 512*1000 = 512000, N=256
    // Block tile: 16x16 => grid dims: ceil(512000/16)=32000, ceil(256/16)=16
    dim3 block(TILE_N, TILE_M); // x for N, y for rows (so threadIdx.x=col, threadIdx.y=row)
    dim3 grid((N + TILE_N - 1) / TILE_N, (B * L + TILE_M - 1) / TILE_M);

    linear_forward_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        (int)B, (int)L, (int)K, (int)N,
        (int)input_stride_b, (int)input_stride_l, (int)input_stride_k,
        (int)weight_stride_n, (int)weight_stride_k,
        (int)output_stride_b, (int)output_stride_l, (int)output_stride_n);
}
