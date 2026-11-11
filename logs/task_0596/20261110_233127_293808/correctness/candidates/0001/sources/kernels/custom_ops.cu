#include <cuda_runtime.h>

// Element-wise absolute value kernel
__global__ void abs_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = fabsf(val);
    }
}

// Pixel shuffle kernel
// Input: [N, C, H, W], C = out_channels * r^2
// Output: [N, out_channels, H*r, W*r]
// Each thread computes one output element.
__global__ void pixel_shuffle_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int64_t N, int64_t C, int64_t H, int64_t W,
                                     int64_t r, int64_t out_channels,
                                     const int64_t* __restrict__ input_strides,
                                     const int64_t* __restrict__ output_strides) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_output = N * out_channels * H * r * W * r;
    int64_t grid_stride = blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total_output; linear += grid_stride) {
        // Decode output linear index to (n, oc, oh, ow)
        int64_t tmp = linear;
        int64_t ow = tmp % (W * r);
        tmp /= (W * r);
        int64_t oh = tmp % (H * r);
        tmp /= (H * r);
        int64_t oc = tmp % out_channels;
        tmp /= out_channels;
        int64_t n = tmp;

        // Compute input channel and spatial coordinates
        int64_t ih = oh / r;
        int64_t iw = ow / r;
        int64_t ic = oc * r * r + (oh % r) * r + (ow % r);

        // Calculate offset using strides for non-contiguous support
        int64_t input_offset = n * input_strides[0] +
                               ic * input_strides[1] +
                               ih * input_strides[2] +
                               iw * input_strides[3];
        int64_t output_offset = n * output_strides[0] +
                                oc * output_strides[1] +
                                oh * output_strides[2] +
                                ow * output_strides[3];

        output[output_offset] = input[input_offset];
    }
}

// Launcher for abs forward
extern "C" void abs_forward_launcher(const float* input,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    // Cap blocks to avoid excessive grid dimensions; kernel uses grid-stride loop
    if (blocks > 65535) blocks = 65535;
    abs_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for pixel shuffle forward
extern "C" void pixel_shuffle_forward_launcher(const float* input,
                                               float* output,
                                               int64_t N, int64_t C, int64_t H, int64_t W,
                                               int64_t r, int64_t out_channels,
                                               const int64_t* input_strides,
                                               const int64_t* output_strides,
                                               cudaStream_t stream) {
    int64_t total_output = N * out_channels * H * r * W * r;
    int threads = 256;
    int blocks = (int)((total_output + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    pixel_shuffle_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, r, out_channels, input_strides, output_strides);
}
