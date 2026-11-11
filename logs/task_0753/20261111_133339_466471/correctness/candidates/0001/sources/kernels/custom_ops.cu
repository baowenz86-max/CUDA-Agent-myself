#include <cuda_runtime.h>
#include <cstdint>

// ---------- Softmax over dim=1 ----------
// Input: x [B, C, L] contiguous float32
// For each batch b and position l, compute softmax over C channels.
// We launch one block per (b, l) pair, with blockDim.x covering C.
// threads: C = 32, so use 32 threads per block, grid = B*L.
// Use shared memory to compute max and sum efficiently, but since C=32,
// we can simply use warp reductions for max and sum. We'll still use
// a straightforward approach: each block computes max, then stores to shared,
// compute sum, then write output.

__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int C, int L) {
    int b = blockIdx.x / L;
    int l = blockIdx.x % L;
    int channel = threadIdx.x;
    if (channel >= C) return;

    const float* x = input + b * C * L + l;
    float val = x[channel * L];

    // Compute max using shared memory reduction
    __shared__ float s_max[32];
    s_max[channel] = val;
    __syncthreads();

    // Reduce within warp (C <= 32, so single warp)
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (channel < offset) {
            float other = s_max[channel + offset];
            if (other > s_max[channel]) s_max[channel] = other;
        }
        __syncthreads();
    }
    float max_val = s_max[0];

    // Compute exp(x - max) and sum
    float exp_val = __expf(val - max_val);
    __shared__ float s_sum[32];
    s_sum[channel] = exp_val;
    __syncthreads();

    for (int offset = 16; offset > 0; offset >>= 1) {
        if (channel < offset) {
            s_sum[channel] += s_sum[channel + offset];
        }
        __syncthreads();
    }
    float sum_val = s_sum[0];

    // Write softmax result
    float* y = output + b * C * L + l;
    y[channel * L] = exp_val / sum_val;
}

// Launcher for softmax
extern "C" void custom_softmax_dim1_launcher(const float* input,
                                              float* output,
                                              int batch,
                                              int channels,
                                              int length,
                                              cudaStream_t stream) {
    int C = channels;
    int L = length;
    int B = batch;

    // Each block handles one (b, l)
    int total_blocks = B * L;
    dim3 grid(total_blocks);
    dim3 block(C);

    softmax_dim1_kernel<<<grid, block, 0, stream>>>(input, output, C, L);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be surfaced by binding via cudaGetLastError after launch;
        // but we can't include caffe. Just leave, but we may want to record.
        // In proper binding, they should call cudaGetLastError after launcher.
        // We cannot include library headers, so we do nothing here.
    }
}

// ---------- Max Pool 1D ----------
// Input: x [B, C, L], contiguous float32
// Parameters: kernel_size, stride, padding, dilation, ceil_mode
// Output length: floor((L + 2*padding - dilation*(kernel_size-1) -1)/stride) + 1
// For each output position i (along length), compute max over j=0..K-1 where
// input_idx = i*stride - padding + j*dilation, if 0 <= input_idx < L.
// We launch one thread per output element, grid-stride loop if needed.

__global__ void max_pool1d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int B, int C, int L_in,
                                  int kernel_size, int stride, int padding, int dilation,
                                  int L_out) {
    int total = B * C * L_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_total = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride_total) {
        int l_out = i % L_out;
        int tmp = i / L_out;
        int c = tmp % C;
        int b = tmp / C;

        int base = (b * C + c) * L_in;
        int start = l_out * stride - padding;
        float max_val = -3.402823466e+38f; // -FLT_MAX

        for (int j = 0; j < kernel_size; ++j) {
            int idx_in = start + j * dilation;
            if (idx_in >= 0 && idx_in < L_in) {
                float val = input[base + idx_in];
                if (val > max_val) max_val = val;
            }
        }
        output[base + l_out] = max_val;
    }
}

// Launcher for max_pool1d
extern "C" void custom_max_pool1d_launcher(const float* input,
                                            float* output,
                                            int batch,
                                            int channels,
                                            int length,
                                            int kernel_size,
                                            int stride,
                                            int padding,
                                            int dilation,
                                            bool ceil_mode,
                                            cudaStream_t stream) {
    int L_in = length;
    int L_out;
    if (ceil_mode) {
        // ceil mode: L_out = ceil((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
        // But PyTorch formula for ceil_mode=True is more complex; however contract states ceil_mode=False.
        // We implement general formula as used in PyTorch for ceil_mode=False:
        // L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
        // For ceil_mode True, use ceil division; but contract uses False, so safe to ignore.
        int numerator = L_in + 2 * padding - dilation * (kernel_size - 1) - 1;
        L_out = (numerator + stride - 1) / stride + 1;
    } else {
        int numerator = L_in + 2 * padding - dilation * (kernel_size - 1) - 1;
        if (numerator < 0) numerator = 0;
        L_out = numerator / stride + 1;
    }
    // Ensure L_out >= 1
    if (L_out < 1) L_out = 1;

    int total = batch * channels * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) {
        // cap grid to avoid limit; use grid-stride loop handles it
        blocks = 65535;
    }

    max_pool1d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, L_in,
        kernel_size, stride, padding, dilation, L_out);

    cudaError_t err = cudaGetLastError();
    // error handled by binding
}
