#include <cuda_runtime.h>

// Kernel: sum over dimension 1 (channels) for 3D tensor (batch, channels, length)
// Output shape: (batch, length)
// Supports arbitrary strides for input.
__global__ void sum_dim1_kernel(float* __restrict__ output, const float* __restrict__ input,
                                int batch, int channels, int length,
                                long long in_stride_batch, long long in_stride_channel, long long in_stride_length) {
    // Each thread computes one output element: (b, l)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * length;
    if (idx >= total) return;

    int b = idx / length;
    int l = idx - b * length;

    float sum = 0.0f;
    const float* base = input + b * in_stride_batch + l * in_stride_length;
    for (int c = 0; c < channels; ++c) {
        sum += base[c * in_stride_channel];
    }
    output[idx] = sum;
}

// Launcher: sum_dim_forward
// input: pointer to input tensor data (float)
// output: pointer to output tensor data (float)
// dim: dimension to reduce (0,1,2 or negative variants), but only dim=1 is optimized
// batch, channels, length: input dimensions (assume 3D tensor)
// in_stride_batch, in_stride_channel, in_stride_length: strides in elements
// out_stride_batch, out_stride_length: strides for output (if output is 2D)
// stream: CUDA stream
// The binding layer will handle allocation and dimension checks.
extern "C" void sum_dim_launcher(float* output, const float* input,
                                 int dim, int batch, int channels, int length,
                                 long long in_stride_batch, long long in_stride_channel, long long in_stride_length,
                                 long long out_stride_batch, long long out_stride_length,
                                 cudaStream_t stream) {
    // Only dim=1 is supported in this kernel; dim handling should be done in binding if needed.
    // The contract specifies dim=1 for this model.
    // We assume dim=1, but we include the parameter for future flexibility.
    if (dim != 1) {
        // For other dims, fallback not implemented in this stage; binding should not call this.
        // Return without action; binding should not rely on this.
        return;
    }

    int total = batch * length;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    sum_dim1_kernel<<<blocks, threads, 0, stream>>>(
        output, input, batch, channels, length,
        in_stride_batch, in_stride_channel, in_stride_length);
}
