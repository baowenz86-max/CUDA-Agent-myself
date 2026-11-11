// CUDA kernel for fused LayerNorm + sum over sequence dimension.
// Computes out[b,f] = sum_s ( (x[b,s,f]-mean[b,s]) / sqrt(var[b,s]+eps) * weight[f] + bias[f] )
// Input: x shape [B,S,F], weight [F], bias [F], output [B,F].
// All tensors are float32. Only CUDA runtime is used.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_DIM 128

// Kernel to compute per-row mean and variance (biased) over last dimension F.
// One block per row, with BLOCK_DIM = F = 128 threads.
// Each thread handles one element of the row (if F <= BLOCK_DIM, exactly one element).
__global__ void compute_row_stats_kernel(const float* __restrict__ x,
                                         float* __restrict__ mean,
                                         float* __restrict__ inv_std,
                                         int B,
                                         int S,
                                         int F,
                                         int stride_b,
                                         int stride_s,
                                         int stride_f,
                                         float eps) {
    int row = blockIdx.x;
    if (row >= B * S) return;

    int b = row / S;
    int s = row % S;
    const float* row_ptr = x + b * stride_b + s * stride_s;

    // Each thread handles one element (or multiple if F > BLOCK_DIM, but here F=128 equal)
    float sum = 0.0f;
    float sum_sq = 0.0f;
    if (threadIdx.x < F) {
        float v = row_ptr[threadIdx.x * stride_f];
        sum = v;
        sum_sq = v * v;
    }

    // Reduction within block using shared memory
    __shared__ float shared_sum[BLOCK_DIM];
    __shared__ float shared_sq[BLOCK_DIM];
    shared_sum[threadIdx.x] = sum;
    shared_sq[threadIdx.x] = sum_sq;
    __syncthreads();

    // Reduce with tree reduction in shared memory
    for (int offset = BLOCK_DIM / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
            shared_sq[threadIdx.x] += shared_sq[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float total_sum = shared_sum[0];
        float total_sq = shared_sq[0];
        float mean_val = total_sum / F;
        float var_val = total_sq / F - mean_val * mean_val;
        mean[row] = mean_val;
        inv_std[row] = rsqrtf(var_val + eps);
    }
}

// Kernel to accumulate output: out[b,f] = sum_s ( (x[b,s,f]-mean[b,s]) * inv_std[b,s] * weight[f] + bias[f] )
// Each thread block computes a tile of output: blockIdx.x -> f index, blockIdx.y -> b index
__global__ void layer_norm_sum_kernel(const float* __restrict__ x,
                                      const float* __restrict__ mean,
                                      const float* __restrict__ inv_std,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      float* __restrict__ out,
                                      int B,
                                      int S,
                                      int F,
                                      int stride_b,
                                      int stride_s,
                                      int stride_f,
                                      int out_stride_b,
                                      int out_stride_f) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (f >= F || b >= B) return;

    float w = weight[f];
    float bias_val = bias[f];
    float acc = bias_val * S; // sum of bias over S
    for (int s = 0; s < S; ++s) {
        const float* row_ptr = x + b * stride_b + s * stride_s;
        int row = b * S + s;
        float norm = (row_ptr[f * stride_f] - mean[row]) * inv_std[row];
        acc += norm * w;
    }
    out[b * out_stride_b + f * out_stride_f] = acc;
}

// Host launcher (C interface, called from binding.cpp)
extern "C" void layer_norm_sum_launcher(const float* x,
                                        const float* weight,
                                        const float* bias,
                                        float* out,
                                        int B,
                                        int S,
                                        int F,
                                        int stride_b,
                                        int stride_s,
                                        int stride_f,
                                        int out_stride_b,
                                        int out_stride_f,
                                        double eps,
                                        cudaStream_t stream) {
    // Allocate temporary arrays for mean and inv_std
    float* d_mean = nullptr;
    float* d_inv_std = nullptr;
    size_t temp_size = (size_t)B * S;
    cudaError_t err = cudaMalloc(&d_mean, temp_size * sizeof(float));
    if (err != cudaSuccess) return;
    err = cudaMalloc(&d_inv_std, temp_size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_mean);
        return;
    }

    // Compute stats: one block per row, 128 threads each
    int gridRows = B * S;
    float eps_f = (float)eps;
    compute_row_stats_kernel<<<gridRows, BLOCK_DIM, 0, stream>>>(
        x, d_mean, d_inv_std, B, S, F, stride_b, stride_s, stride_f, eps_f);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_mean);
        cudaFree(d_inv_std);
        return;
    }

    // Accumulate output
    dim3 blockOut(16, 16);
    dim3 gridOut((F + blockOut.x - 1) / blockOut.x, B);
    layer_norm_sum_kernel<<<gridOut, blockOut, 0, stream>>>(
        x, d_mean, d_inv_std, weight, bias, out,
        B, S, F, stride_b, stride_s, stride_f, out_stride_b, out_stride_f);
    err = cudaGetLastError();

    cudaFree(d_mean);
    cudaFree(d_inv_std);
    (void)err; // caller may check
}