// CUDA kernels for custom_linear_forward, custom_silu_forward, custom_permute_forward
#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Linear forward kernel: output[b,c,h,o] = sum_k input[b,c,h,k] * weight[o,k] + bias[o]
// Input tensor shape: [batch, in_features, spatial0, spatial1] (spatial0=height, spatial1=width)
// weight shape: [out_features, in_features]
// bias shape: [out_features]
// Strides are provided to support non-contiguous input.
// output is contiguous with shape [batch, in_features, spatial0, out_features]
// Thread mapping: one thread per output element (b, c, h, o)
// Grid: (out_features, batch * in_features * spatial0) // block (256)
// Actually we use 1D grid over total output elements: batch*in_features*spatial0*out_features
// Each thread computes one output element with reduction over in_features.
// This is memory bandwidth bound but correctness is priority for now.
// block size 256, grid-stride loop handles large sizes.
__global__ void linear_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_features,
    int spatial0,
    int spatial1,
    int out_features,
    long long input_stride_batch,   // stride between batches (usually in_features*spatial0*spatial1)
    long long input_stride_channel, // stride between channels (usually spatial0*spatial1)
    long long input_stride_spatial0,// stride between spatial0 elements (usually spatial1)
    long long input_stride_spatial1,// stride between spatial1 elements (usually 1)
    long long weight_stride_out,    // stride for weight rows (usually in_features)
    long long weight_stride_in,     // stride for weight columns (usually 1)
    long long output_stride_batch,  // output strides for contiguity
    long long output_stride_channel,
    long long output_stride_spatial0,
    long long output_stride_out
) {
    long long total = (long long)batch * in_features * spatial0 * out_features;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        // Decompose idx into b, c, h, o
        int o = idx % out_features;
        long long tmp = idx / out_features;
        int h = tmp % spatial0;
        tmp /= spatial0;
        int c = tmp % in_features;
        int b = tmp / in_features;

        // Compute base input address for this (b, c, h)
        const float* base_input = input
            + b * input_stride_batch
            + c * input_stride_channel
            + h * input_stride_spatial0;

        float sum = 0.0f;
        // Calculate dot product over k (spatial1 dimension)
        #pragma unroll 4
        for (int k = 0; k < spatial1; ++k) {
            float in_val = base_input[k * input_stride_spatial1];
            float w_val = weight[o * weight_stride_out + k * weight_stride_in];
            sum += in_val * w_val;
        }

        if (bias != nullptr) {
            sum += bias[o];
        }

        // Write output at computed location
        long long out_idx = ((long long)b * output_stride_batch +
                             (long long)c * output_stride_channel +
                             (long long)h * output_stride_spatial0 +
                             (long long)o * output_stride_out);
        output[out_idx] = sum;
    }
}

// Launcher for linear_forward
// Parameters: input, weight, bias pointers (bias may be null)
// dims: batch, in_features, spatial0, spatial1, out_features
// strides: input strides in elements (not bytes) for batch, channel, spatial0, spatial1
// weight strides: out stride, in stride
// output strides: batch, channel, spatial0, out (all contiguous typically)
// stream
extern "C" void custom_linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_features,
    int spatial0,
    int spatial1,
    int out_features,
    long long input_stride_batch,
    long long input_stride_channel,
    long long input_stride_spatial0,
    long long input_stride_spatial1,
    long long weight_stride_out,
    long long weight_stride_in,
    long long output_stride_batch,
    long long output_stride_channel,
    long long output_stride_spatial0,
    long long output_stride_out,
    cudaStream_t stream
) {
    long long total = (long long)batch * in_features * spatial0 * out_features;
    if (total == 0) return;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    // Cap grid to reasonable maximum to avoid too many blocks; use grid-stride loop if needed
    int max_grid = 65535;
    if (grid_size > max_grid) grid_size = max_grid;

    linear_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, output,
        batch, in_features, spatial0, spatial1, out_features,
        input_stride_batch, input_stride_channel, input_stride_spatial0, input_stride_spatial1,
        weight_stride_out, weight_stride_in,
        output_stride_batch, output_stride_channel, output_stride_spatial0, output_stride_out
    );
}

// -----------------------------------------------------------------------------
// SiLU forward: output[i] = input[i] * sigmoid(input[i])
// Supports arbitrary strides: each thread processes one element using linear index.
__global__ void silu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel,
    long long* input_strides,   // array of 4 strides (batch, channel, spatial0, spatial1)
    int ndim // 4
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    // For simplicity, assume 4D and pass precomputed base offset and strides.
    // However, to avoid kernel with array parameter, we can specialize for 4D.
    // Since ndim is always 4 from contract, we can hardcode 4D indexing.
    // But input may be non-contiguous; we need strides.
    // We'll pass strides as four long longs.
}

// Better to declare a kernel with explicit strides for 4D
__global__ void silu_forward_kernel_4d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channel,
    int spatial0,
    int spatial1,
    long long stride_batch,
    long long stride_channel,
    long long stride_spatial0,
    long long stride_spatial1
) {
    long long total = (long long)batch * channel * spatial0 * spatial1;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += grid_stride) {
        int w = idx % spatial1;
        long long tmp = idx / spatial1;
        int h = tmp % spatial0;
        tmp /= spatial0;
        int c = tmp % channel;
        int b = tmp / channel;

        long long offset = (long long)b * stride_batch +
                          (long long)c * stride_channel +
                          (long long)h * stride_spatial0 +
                          (long long)w * stride_spatial1;
        float x = input[offset];
        // sigmoid(x) = 1/(1+exp(-x))
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;  // output is contiguous, idx is linear index
    }
}

extern "C" void custom_silu_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channel,
    int spatial0,
    int spatial1,
    long long stride_batch,
    long long stride_channel,
    long long stride_spatial0,
    long long stride_spatial1,
    cudaStream_t stream
) {
    long long total = (long long)batch * channel * spatial0 * spatial1;
    if (total == 0) return;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    int max_grid = 65535;
    if (grid_size > max_grid) grid_size = max_grid;

    silu_forward_kernel_4d<<<grid_size, block_size, 0, stream>>>(
        input, output, batch, channel, spatial0, spatial1,
        stride_batch, stride_channel, stride_spatial0, stride_spatial1
    );
}

// -----------------------------------------------------------------------------
// Permute forward: copy input to output with permuted dimensions.
// Here permute dims are (0,2,3,1), so output shape is [batch, spatial0, spatial1, channel].
// We assume dims fixed for this contract. The launcher takes dims array but kernel implements (0,2,3,1).
// For generality, could pass permute indices, but contract only requires this case.
// Therefore, implement a kernel that performs (0,2,3,1) permutation.
// Output strides are typically contiguous, but we can write linearly.
__global__ void permute_forward_kernel_0231(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channel,
    int spatial0,
    int spatial1,
    long long input_stride_batch,
    long long input_stride_channel,
    long long input_stride_spatial0,
    long long input_stride_spatial1
) {
    // Output layout: [batch, spatial0, spatial1, channel]
    // Linear index in output: ((b * spatial0 + h) * spatial1 + w) * channel + c
    long long total = (long long)batch * spatial0 * spatial1 * channel;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += grid_stride) {
        int c = idx % channel;
        long long tmp = idx / channel;
        int w = tmp % spatial1;
        tmp /= spatial1;
        int h = tmp % spatial0;
        int b = tmp / spatial0;

        long long input_offset = (long long)b * input_stride_batch +
                                (long long)c * input_stride_channel +
                                (long long)h * input_stride_spatial0 +
                                (long long)w * input_stride_spatial1;
        output[idx] = input[input_offset];
    }
}

extern "C" void custom_permute_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channel,
    int spatial0,
    int spatial1,
    long long input_stride_batch,
    long long input_stride_channel,
    long long input_stride_spatial0,
    long long input_stride_spatial1,
    cudaStream_t stream
) {
    long long total = (long long)batch * spatial0 * spatial1 * channel;
    if (total == 0) return;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    int max_grid = 65535;
    if (grid_size > max_grid) grid_size = max_grid;

    permute_forward_kernel_0231<<<grid_size, block_size, 0, stream>>>(
        input, output, batch, channel, spatial0, spatial1,
        input_stride_batch, input_stride_channel, input_stride_spatial0, input_stride_spatial1
    );
}
