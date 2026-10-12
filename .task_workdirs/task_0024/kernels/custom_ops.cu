#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// ============================================================================
// ConvTranspose1d forward kernel
// ============================================================================

__global__ void conv_transpose1d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding)
{
    int total_elements = batch_size * out_channels * out_length;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;

    for (int linear_idx = idx; linear_idx < total_elements; linear_idx += grid_stride) {
        // Decode linear index to (n, oc, ow)
        int ow = linear_idx % out_length;
        int oc = (linear_idx / out_length) % out_channels;
        int n = linear_idx / (out_length * out_channels);

        float acc = 0.0f;

        // Convert output position to input position: iw * stride = ow + padding - k
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                int numerator = ow + padding - k;
                if (numerator < 0) continue;
                if (numerator % stride != 0) continue;
                int iw = numerator / stride;
                if (iw >= in_length) continue;

                // weight shape: [in_channels, out_channels, kernel_size]
                float w = weight[(ic * out_channels + oc) * kernel_size + k];
                float x_val = input[(n * in_channels + ic) * in_length + iw];
                acc += w * x_val;
            }
        }

        // Add bias if provided (raw pointer may be null, but launcher should pass valid pointer)
        // We assume bias is always provided; if not, pass a zero-filled tensor from binding.
        if (bias != nullptr) {
            acc += bias[oc];
        }

        output[linear_idx] = acc;
    }
}

// ============================================================================
// Median kernel (global median of arbitrary size)
// ============================================================================

__global__ void copy_to_scratch_kernel(
    const float* __restrict__ input,
    float* __restrict__ scratch,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        scratch[idx] = input[idx];
    }
}

// ============================================================================
// Element-wise not-equal kernels: float vs float scalar
// ============================================================================

__global__ void ne_float_scalar_kernel(
    const float* __restrict__ input,
    float scalar,
    bool* __restrict__ output,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;

    for (int i = idx; i < total_elements; i += grid_stride) {
        output[i] = (input[i] != scalar);
    }
}

// ============================================================================
// Host launchers (extern "C" for binding.cpp)
// ============================================================================

extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream)
{
    int total_elements = batch_size * out_channels * out_length;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // grid-stride loop handles large data
    if (blocks > 65535) {
        blocks = 65535; // max grid dim for x
    }

    conv_transpose1d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels, in_length, out_length,
        kernel_size, stride, padding, output_padding);
}

extern "C" void median_launcher(
    const float* input,
    float* output,
    int total_elements,
    cudaStream_t stream)
{
    if (total_elements <= 0) {
        // Should not happen, but set output to NaN or 0? Reference torch.median would error.
        // But binding ensures non-empty, so we set 0.0f.
        cudaMemsetAsync(output, 0, sizeof(float), stream);
        return;
    }

    // Allocate temporary scratch buffer for sorting
    float* d_scratch = nullptr;
    cudaMalloc(&d_scratch, (size_t)total_elements * sizeof(float));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error by setting output to 0? But better to throw in binding, but here we just write.
        // Since binding should have checked, this is unlikely.
        cudaMemsetAsync(output, 0, sizeof(float), stream);
        return;
    }

    // Copy input to scratch
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    copy_to_scratch_kernel<<<blocks, threads, 0, stream>>>(input, d_scratch, total_elements);

    // Sort scratch using thrust
    thrust::sort(thrust::cuda::par.on(stream), d_scratch, d_scratch + total_elements);

    // Determine the median index: lower median for even count
    int median_index = (total_elements - 1) / 2; // integer floor
    // For even N, e.g., N=4 -> (4-1)/2 =1 (0-based), which is the lower median.
    // For odd N, e.g., N=5 -> (5-1)/2 =2, middle.

    // Copy median value to output
    cudaMemcpyAsync(output, d_scratch + median_index, sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // Free scratch asynchronously? Need to free after copy completes.
    // We can use cudaFree after stream sync, but here we can't sync. Instead we can use cudaFreeAsync? No such.
    // Better to schedule cudaFree after stream operations? cudaFree is synchronous on current context.
    // For simplicity, we use cudaFree(d_scratch) which will synchronize implicitly (unless we use many streams).
    // Since this launcher is called per forward, overhead may be large. We can use cudaMallocAsync with pool,
    // but for simplicity we use cudaFree after kernel launches; it will defer until stream completes? Actually cudaFree
    // synchronizes on the current stream? It may synchronize the entire device, but acceptable for correctness.
    cudaFree(d_scratch);
}

extern "C" void ne_float_scalar_launcher(
    const float* input,
    float scalar,
    bool* output,
    int total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) {
        blocks = 65535;
    }
    ne_float_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input, scalar, output, total_elements);
}
