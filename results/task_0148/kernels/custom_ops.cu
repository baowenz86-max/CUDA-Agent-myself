#include <cuda_runtime.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// ReflectionPad2d with padding (1,1,1,1) for float32 input.
// Input tensor shape: [N, C, H, W], output on device.
// Strides are passed explicitly to support non-contiguous input.
// Kernel computes output element (n,c,out_h,out_w) by mapping to input index.
// Reflection formula: idx = mirror around edges without repeating edge.
// For pad=1: if out_h==0 => in_h=1; if out_h==H+1 => in_h=H-2; else in_h=out_h-1.
// Similarly for width.
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_H * out_W;
    if (idx >= total) return;

    int w = idx % out_W;
    int tmp = idx / out_W;
    int h = tmp % out_H;
    tmp /= out_H;
    int c = tmp % C;
    int n = tmp / C;

    // Map output spatial coordinates to input coordinates with reflection.
    int in_h, in_w;
    if (h == 0) in_h = 1;
    else if (h == out_H - 1) in_h = H - 2;
    else in_h = h - 1;

    if (w == 0) in_w = 1;
    else if (w == out_W - 1) in_w = W - 2;
    else in_w = w - 1;

    // Clamp to valid range (should be valid for pad=1 and H,W>=2)
    in_h = min(max(in_h, 0), H - 1);
    in_w = min(max(in_w, 0), W - 1);

    const float* in_ptr = input + n * in_n_stride + c * in_c_stride + in_h * in_h_stride + in_w * in_w_stride;
    float* out_ptr = output + n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride;
    *out_ptr = *in_ptr;
}

// -----------------------------------------------------------------------------
// Softmax over channel dimension (dim=1) for each spatial location (n, h, w).
// Input shape: [N, C, H, W], output same shape.
// Uses online softmax for numerical stability and better memory access pattern.
// Each thread block handles a set of spatial positions and channels.
// We'll use one block per (n, h, w) but batch spatial positions for better occupancy.
__global__ void softmax_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride)
{
    // Each block handles a single (n, h, w) vector across channels.
    int block_idx = blockIdx.x;
    int n = block_idx / (H * W);
    int rem = block_idx % (H * W);
    int h = rem / W;
    int w = rem % W;

    // Number of channels C is small (3), use shared memory for values.
    extern __shared__ float smem[];  // size C floats for values, plus reduction of max and sum
    float* vals = smem;
    float* max_val = &smem[C];
    float* sum_val = &smem[C + 1];
    int tid = threadIdx.x;

    // Load input values into shared memory (coalesced across channels? better to use only one thread per channel? C=3, blockDim may be 32, only first C threads active)
    if (tid < C) {
        vals[tid] = input[n * in_n_stride + tid * in_c_stride + h * in_h_stride + w * in_w_stride];
    }
    __syncthreads();

    // Reduce max (online softmax style uses running max, but C small => simple reduction)
    if (tid == 0) {
        float m = vals[0];
        for (int i = 1; i < C; ++i) {
            m = fmaxf(m, vals[i]);
        }
        *max_val = m;

        float s = 0.0f;
        for (int i = 0; i < C; ++i) {
            s += __expf(vals[i] - m);
        }
        *sum_val = s;
    }
    __syncthreads();

    // Write output using shared max and sum.
    if (tid < C) {
        float out_val = __expf(vals[tid] - *max_val) / *sum_val;
        output[n * out_n_stride + tid * out_c_stride + h * out_h_stride + w * out_w_stride] = out_val;
    }
}

// -----------------------------------------------------------------------------
// Histogram over all elements: counts values in [min, max] into bins equal-width bins.
// Input shape: [N, C, H, W], arbitrary strides.
// Output: bins float32 counts, each count potentially large (up to ~50M). Use float32 as required, but accumulation can be done via integer counts then convert.
// We'll use atomicAdd on float for simplicity; but float atomic may have precision issues. Better use unsigned int atomics and convert later. Since max count ~50M fits in 32-bit int, use atomicAdd on int, then convert to float in output.
__global__ void histc_float_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int numel,
    int bins,
    float min_val,
    float max_val,
    int n_stride, int c_stride, int h_stride, int w_stride,
    int N, int C, int H, int W)
{
    extern __shared__ int tmp_counts[]; // bins integers
    int tid = threadIdx.x;
    if (tid < bins) tmp_counts[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    while (idx < numel) {
        // Compute coordinates from linear index using strides (non-contiguous)
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;
        float val = input[n * n_stride + c * c_stride + h * h_stride + w * w_stride];
        if (val >= min_val && val <= max_val) {
            float scaled = (val - min_val) * (float)bins / (max_val - min_val);
            int bin = (int)scaled;
            if (bin == bins) bin = bins - 1; // handle val == max_val edge
            if (bin >= 0 && bin < bins) {
                atomicAdd(&tmp_counts[bin], 1);
            }
        }
        idx += blockDim.x * gridDim.x; // grid-stride
    }
    __syncthreads();

    // Reduce block partial counts into global output (float) via atomicAdd on floats (converted from int)
    // But output is float, we need to add integer counts to float; use atomicAdd on float with (float)count.
    if (tid < bins) {
        atomicAdd(&output[tid], (float)tmp_counts[tid]);
    }
}

// Launcher functions (extern "C" for binding)
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
    cudaStream_t stream)
{
    int total = N * C * out_H * out_W;
    int block = 256;
    int grid = (total + block - 1) / block;
    reflection_pad2d_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, H, W, out_H, out_W,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Could handle error; but not in .cu? We can ignore or use printf for debug (not recommended).
    }
}

extern "C" void softmax_dim1_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
    cudaStream_t stream)
{
    int num_blocks = N * H * W;
    // Block size: use next multiple of 32 >= C? Better use 32 or 64 threads, but only first C threads active. Since C=3, 32 is fine.
    int block = 32; // minimal warp
    // Shared memory: C floats for values + 1 for max + 1 for sum = C+2 floats
    int shared_size = (C + 2) * sizeof(float);
    softmax_dim1_kernel<<<num_blocks, block, shared_size, stream>>>(
        input, output, N, C, H, W,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride);
    cudaGetLastError();
}

extern "C" void histc_launcher(
    const float* input,
    float* output,
    int numel,
    int bins,
    float min_val,
    float max_val,
    int n_stride, int c_stride, int h_stride, int w_stride,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    // Zero output first (binding should have done that via torch::zeros, but maybe safer to cudaMemset? binding will allocate zero, but to be safe, we can cudaMemsetAsync)
    cudaMemsetAsync(output, 0, bins * sizeof(float), stream);
    // Launch kernel with grid-stride loop and shared memory partial sums.
    int block = 256;
    int grid = 1024; // tune? Use enough blocks to saturate, but grid-stride loop inside.
    int shared_size = bins * sizeof(int);
    histc_float_kernel<<<grid, block, shared_size, stream>>>(
        input, output, numel, bins, min_val, max_val,
        n_stride, c_stride, h_stride, w_stride,
        N, C, H, W);
    cudaGetLastError();
}
