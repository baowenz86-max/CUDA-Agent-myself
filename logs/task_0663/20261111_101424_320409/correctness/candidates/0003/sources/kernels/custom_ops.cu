#include <cuda_runtime.h>

// Helper to compute index offset given linear index and strides/dims, handling stride-0 broadcast
__device__ __forceinline__ int64_t compute_offset(
    int64_t linear_idx,
    const int64_t* dims,
    const int64_t* strides,
    int ndim) {
    int64_t offset = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t stride = strides[d];
        int64_t dim = dims[d];
        int64_t idx = linear_idx % dim;
        linear_idx /= dim;
        if (stride != 0) {
            offset += idx * stride;
        }
    }
    return offset;
}

// ReLU6 elementwise kernel, supports arbitrary strides
__global__ void relu6_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements,
    int ndim,
    const int64_t* __restrict__ dims,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_strides) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        int64_t in_offset = compute_offset(idx, dims, input_strides, ndim);
        int64_t out_offset = compute_offset(idx, dims, output_strides, ndim);
        float val = input[in_offset];
        val = val > 6.0f ? 6.0f : (val < 0.0f ? 0.0f : val);
        output[out_offset] = val;
    }
}

// Sum over dimension 1 (channels), output has shape [B,1,H,W]
// We parallelize over output elements: each thread computes sum for one (b,1,h,w)
__global__ void sum_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* __restrict__ input_strides, // input stride array [B_stride, C_stride, H_stride, W_stride]
    const int64_t* __restrict__ output_strides) { // output stride array [B_stride, 1_stride, H_stride, W_stride]

    int64_t total = B * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int64_t w = idx % W;
        int64_t tmp = idx / W;
        int64_t h = tmp % H;
        int64_t b = tmp / H;

        float sum = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            int64_t in_offset = b * input_strides[0] + c * input_strides[1] +
                                h * input_strides[2] + w * input_strides[3];
            sum += input[in_offset];
        }
        int64_t out_offset = b * output_strides[0] + h * output_strides[2] + w * output_strides[3];
        output[out_offset] = sum;
    }
}

// Expand kernel: replicate input along dimension 1.
// Input shape is [B,1,H,W], output is [B,out_features,H,W] with output_stride[1] possibly 0 (broadcast).
// We iterate over output elements; for each output element we compute input offset using broadcast rules on the second dimension.
__global__ void expand_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t B,
    int64_t out_features,
    int64_t H,
    int64_t W,
    const int64_t* __restrict__ input_strides,  // [s_B, s_1, s_H, s_W] for input shape [B,1,H,W], s_1 can be anything but dim=1
    const int64_t* __restrict__ output_strides) {

    int64_t total = B * out_features * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int64_t w = idx % W;
        int64_t tmp = idx / W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t b = tmp / out_features;
        int64_t in_offset = b * input_strides[0] + h * input_strides[2] + w * input_strides[3];
        int64_t out_offset = b * output_strides[0] + (idx / (H * W)) % out_features * output_strides[1] +
                             h * output_strides[2] + w * output_strides[3];
        output[out_offset] = input[in_offset];
    }
}

// Launcher for ReLU6
extern "C" void relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    int ndim,
    const int64_t* dims,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream) {

    int threads = 256;
    int64_t blocks64 = (total_elements + threads - 1) / threads;
    int blocks = (blocks64 > 2147483647) ? 2147483647 : (int)blocks64;
    relu6_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total_elements, ndim, dims, input_strides, output_strides);
}

// Launcher for sum_dim1
extern "C" void sum_dim1_forward_launcher(
    const float* input,
    float* output,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream) {

    int threads = 256;
    int64_t total = B * H * W;
    int64_t blocks64 = (total + threads - 1) / threads;
    int blocks = (blocks64 > 2147483647) ? 2147483647 : (int)blocks64;
    sum_dim1_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, C, H, W, input_strides, output_strides);
}

// Launcher for expand
extern "C" void expand_forward_launcher(
    const float* input,
    float* output,
    int64_t B,
    int64_t out_features,
    int64_t H,
    int64_t W,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream) {

    int threads = 256;
    int64_t total = B * out_features * H * W;
    int64_t blocks64 = (total + threads - 1) / threads;
    int blocks = (blocks64 > 2147483647) ? 2147483647 : (int)blocks64;
    expand_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, out_features, H, W, input_strides, output_strides);
}
