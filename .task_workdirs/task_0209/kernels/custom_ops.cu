#include <cuda_runtime.h>

// Elementwise addcmul: output = input + tensor1 * tensor2
// All tensors are 3D with shape (batch, out_channels, out_length)
// Supports arbitrary strides (non-contiguous).
__global__ void addcmul_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ tensor1,
    const float* __restrict__ tensor2,
    int batch,
    int channels,
    int length,
    long long out_stride_batch,
    long long out_stride_channel,
    long long in_stride_batch,
    long long in_stride_channel,
    long long t1_stride_batch,
    long long t1_stride_channel,
    long long t2_stride_batch,
    long long t2_stride_channel
) {
    long long total = (long long)batch * channels * length;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        // decompose linear index into (b, c, l)
        int l = idx % length;
        long long tmp = idx / length;
        int c = tmp % channels;
        int b = tmp / channels;

        long long out_offset = b * out_stride_batch + c * out_stride_channel + l;
        long long in_offset = b * in_stride_batch + c * in_stride_channel + l;
        long long t1_offset = b * t1_stride_batch + c * t1_stride_channel + l;
        long long t2_offset = b * t2_stride_batch + c * t2_stride_channel + l;

        output[out_offset] = input[in_offset] + tensor1[t1_offset] * tensor2[t2_offset];
    }
}

extern "C" void addcmul_launcher(
    float* output,
    const float* input,
    const float* tensor1,
    const float* tensor2,
    int batch,
    int channels,
    int length,
    long long out_stride_batch,
    long long out_stride_channel,
    long long in_stride_batch,
    long long in_stride_channel,
    long long t1_stride_batch,
    long long t1_stride_channel,
    long long t2_stride_batch,
    long long t2_stride_channel,
    cudaStream_t stream
) {
    const int threads = 256;
    long long total = (long long)batch * channels * length;
    long long blocks_ll = (total + threads - 1) / threads;
    // Cap grid to reasonable maximum (e.g., 2^31-1) but grid-stride handles large sizes
    int blocks = (int)min(blocks_ll, (long long)2147483647);
    if (blocks == 0) blocks = 1; // for empty tensor, launch one block that does nothing

    addcmul_kernel<<<blocks, threads, 0, stream>>>(
        output, input, tensor1, tensor2,
        batch, channels, length,
        out_stride_batch, out_stride_channel,
        in_stride_batch, in_stride_channel,
        t1_stride_batch, t1_stride_channel,
        t2_stride_batch, t2_stride_channel
    );
}
