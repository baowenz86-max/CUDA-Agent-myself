#include <cuda_runtime.h>
#include <cstdio>

// LayerNorm over last dimension (H) for each (B,S) pair.
// input: shape [B, S, H], strides: batch_stride, row_stride, col_stride.
// weight, bias optional (may be nullptr).
// output: same shape as input, contiguous.
__global__ void layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int S, int H,
    long long input_batch_stride,
    long long input_row_stride,
    long long input_col_stride,
    double eps)
{
    int row = blockIdx.x; // total rows = B*S
    if (row >= B * S) return;
    int b = row / S;
    int s = row % S;
    const float* row_ptr = input + b * input_batch_stride + s * input_row_stride;
    float* out_row_ptr = output + row * H; // output contiguous: shape [B*S, H]

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        sum += row_ptr[i * input_col_stride];
    }
    // Shared memory reduction for sum (block size <= 1024)
    __shared__ float s_sum[1024]; // enough
    int tid = threadIdx.x;
    s_sum[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }
    float mean = s_sum[0] / H;

    // Compute variance: E[x^2] - mean^2
    float sq_sum = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float x = row_ptr[i * input_col_stride];
        sq_sum += x * x;
    }
    __shared__ float s_sq_sum[1024];
    s_sq_sum[tid] = sq_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sq_sum[tid] += s_sq_sum[tid + offset];
        }
        __syncthreads();
    }
    float var = s_sq_sum[0] / H - mean * mean;
    float inv_std = rsqrtf(var + (float)eps);

    // Normalize and apply affine
    for (int i = tid; i < H; i += blockDim.x) {
        float x = row_ptr[i * input_col_stride];
        float y = (x - mean) * inv_std;
        if (weight != nullptr) y *= weight[i];
        if (bias != nullptr) y += bias[i];
        out_row_ptr[i] = y;
    }
}

// Mean over dimension 1 (sequence length S) for each batch b and hidden h.
// input: shape [B, S, H], strides as above (but likely contiguous).
// output: shape [B, H], contiguous.
__global__ void mean_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int S, int H,
    long long input_batch_stride,
    long long input_row_stride,
    long long input_col_stride)
{
    int out_idx = blockIdx.x; // b*H + h
    if (out_idx >= B * H) return;
    int b = out_idx / H;
    int h = out_idx - b * H;

    float sum = 0.0f;
    int tid = threadIdx.x;
    // Grid-stride over S
    for (int s = tid; s < S; s += blockDim.x) {
        const float* elem_ptr = input + b * input_batch_stride + s * input_row_stride + h * input_col_stride;
        sum += *elem_ptr;
    }
    // Shared memory reduction
    __shared__ float s_sum[1024];
    s_sum[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[out_idx] = s_sum[0] / S;
    }
}

// Launcher for LayerNorm
extern "C" void layer_norm_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int S, int H, double eps,
    cudaStream_t stream)
{
    int rows = B * S;
    dim3 block(256);
    dim3 grid(rows);
    // Assume input is contiguous (stride: row_stride=H, col_stride=1, batch_stride=S*H)
    // But we support non-contiguous via strides: compute from shape assuming row-major.
    long long input_batch_stride = 1LL * S * H;
    long long input_row_stride = H;
    long long input_col_stride = 1;
    layer_norm_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        B, S, H,
        input_batch_stride, input_row_stride, input_col_stride,
        eps);
}

// Launcher for mean over dim=1
extern "C" void mean_dim_forward_launcher(
    const float* input,
    float* output,
    int B, int S, int H,
    cudaStream_t stream)
{
    int out_size = B * H;
    dim3 block(256);
    dim3 grid(out_size);
    long long input_batch_stride = 1LL * S * H;
    long long input_row_stride = H;
    long long input_col_stride = 1;
    mean_dim_kernel<<<grid, block, 0, stream>>>(
        input, output,
        B, S, H,
        input_batch_stride, input_row_stride, input_col_stride);
}
