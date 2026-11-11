#include <cuda_runtime.h>

// Elementwise minimum with broadcasting over channel dimension.
// Input tensors: x has shape [batch, out_channels, length_out] (3D, strides arbitrary)
// y has shape [out_channels, 1] and is broadcast along batch and length dimensions.
// Output has same shape as x, float32.
__global__ void minimum_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    long long batch,
    long long channels,
    long long length,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    long long y_channel_stride,
    long long out_batch_stride,
    long long out_channel_stride,
    long long out_length_stride)
{
    long long total = batch * channels * length;
    // grid-stride loop over linear index in logical (b, c, l) order
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        long long l = idx % length;
        long long tmp = idx / length;
        long long c = tmp % channels;
        long long b = tmp / channels;

        const float x_val = x[b * x_batch_stride + c * x_channel_stride + l * x_length_stride];
        // y is [channels, 1], stride for channel dim is y_channel_stride, length index 0
        const float y_val = y[c * y_channel_stride];
        out[b * out_batch_stride + c * out_channel_stride + l * out_length_stride] =
            fminf(x_val, y_val);
    }
}

// Elementwise floor.
__global__ void floor_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    long long total,
    long long batch_stride,
    long long channel_stride,
    long long length_stride)
{
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        // Indexing not needed for contiguous output? We need to map linear index to 3D using strides.
        // Actually output is same shape and same stride as input. So use out pointer with same linear index but strides.
        // Since output allocated with same shape but potentially contiguous? We cannot assume output contiguous.
        // In launcher we allocate contiguous output, but contract says supports strides for x, output should match x shape and likely contiguous.
        // For safety, we receive input strides and output strides separately? Better to use same logical indexing and output strides passed.
        // To keep launcher simple, output is contiguous and has known strides, but we pass them anyway.
        long long total_elements = total;
        // For each linear idx, compute b,c,l using strides? The linear index assumes row-major contiguous layout.
        // However if input is non-contiguous, we cannot simply map linear idx to 3D by division if strides not compatible.
        // We need to iterate over logical coordinates directly. The grid-stride loop over total elements is only valid if the tensor is contiguous in memory.
        // But contract says supports strides for x and y, so we need proper handling.
        // We'll change approach: use 3D grid? But grid dimension limited. Use block per element with 3D indexing? Could use 3D grid.
        // Simpler: use 1D grid-stride over total logical elements, but compute offsets using actual strides.
        // We need to decode idx into (b,c,l) assuming logical shape. That works regardless of memory layout.
        // So we keep same method as minimum_kernel.
        long long l = idx % length_stride; // WRONG: need actual length, not stride. We'll fix below.
        // Actually the kernel signature should include length, batch, channels, not just total.
    }
}

// Correct floor kernel with shape info and strides.
__global__ void floor_kernel_v2(
    const float* __restrict__ x,
    float* __restrict__ out,
    long long batch,
    long long channels,
    long long length,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    long long out_batch_stride,
    long long out_channel_stride,
    long long out_length_stride)
{
    long long total = batch * channels * length;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        long long l = idx % length;
        long long tmp = idx / length;
        long long c = tmp % channels;
        long long b = tmp / channels;

        const float x_val = x[b * x_batch_stride + c * x_channel_stride + l * x_length_stride];
        out[b * out_batch_stride + c * out_channel_stride + l * out_length_stride] = floorf(x_val);
    }
}

// Launcher for custom_minimum: out = min(x, y) with broadcasting.
extern "C" void custom_minimum_launcher(
    const float* x,
    const float* y,
    float* out,
    long long batch,
    long long channels,
    long long length,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    long long y_channel_stride,
    long long out_batch_stride,
    long long out_channel_stride,
    long long out_length_stride,
    cudaStream_t stream)
{
    long long total = batch * channels * length;
    if (total <= 0) return;
    const int block_size = 256;
    long long blocks = (total + block_size - 1) / block_size;
    // Cap grid to reasonable size and use grid-stride loop
    if (blocks > 65535) blocks = 65535;
    minimum_kernel<<<(int)blocks, block_size, 0, stream>>>(
        x, y, out,
        batch, channels, length,
        x_batch_stride, x_channel_stride, x_length_stride,
        y_channel_stride,
        out_batch_stride, out_channel_stride, out_length_stride);
}

// Launcher for custom_floor: out = floor(x).
extern "C" void custom_floor_launcher(
    const float* x,
    float* out,
    long long batch,
    long long channels,
    long long length,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    long long out_batch_stride,
    long long out_channel_stride,
    long long out_length_stride,
    cudaStream_t stream)
{
    long long total = batch * channels * length;
    if (total <= 0) return;
    const int block_size = 256;
    long long blocks = (total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    floor_kernel_v2<<<(int)blocks, block_size, 0, stream>>>(
        x, out,
        batch, channels, length,
        x_batch_stride, x_channel_stride, x_length_stride,
        out_batch_stride, out_channel_stride, out_length_stride);
}

// Note: custom_conv_transpose1d is intentionally not defined here.
// It will be provided by the third-stage C++ binding using cuDNN.
