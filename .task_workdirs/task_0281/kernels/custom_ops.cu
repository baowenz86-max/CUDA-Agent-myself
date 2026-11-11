#include <cuda_runtime.h>

// Relu forward kernel: output = max(0, input)
// Supports arbitrary strides via input/output pointer arithmetic.
__global__ void relu_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int numel,
    const int* __restrict__ in_strides,
    const int* __restrict__ out_strides,
    int ndim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // Compute linear index to multi-dimensional coordinates
    int linear_idx = idx;
    int in_offset = 0;
    int out_offset = 0;
    int remaining = linear_idx;

    // Assume ndim <= 3 for this task (N,C,L)
    for (int d = 0; d < ndim; ++d) {
        int dim_size = 0;
        // We need dimension sizes to compute coordinates. Since we only have strides, we cannot recover dim sizes directly.
        // Instead, we can compute coordinates using the total size and strides if we know the shape. But we are not given shape in the kernel.
        // The launcher should pass shape as well, or we can compute coordinates by assuming contiguous input for now? 
        // However, contract requires supporting strides, so we need shape.
        // Better approach: use the strides to map linear index to offsets without knowing shape by using the stride values directly.
        // For a tensor with shape [d0,d1,d2] and strides [s0,s1,s2], linear index = i0*s0 + i1*s1 + i2*s2.
        // But to compute i0,i1,i2 we need shape. So kernel must receive shape.
    }

    // The above loop is incomplete. We'll instead create a specialized kernel for 3D tensors (N,C,L) which is our known shape.
    // But the contract says shape can be ([N, C, L]) but we don't know exact N,C,L values at compile time. So launcher will pass strides and also total elements. We can use a mapping that works generically if we know the number of elements along each stride dimension.
    // For simplicity, since support for arbitrary strides is required, we'll pass shape as int array.
    // We'll implement a generic indexing using shape and strides.
}

// Proper implementation: We'll use a generic kernel that takes shape, strides, and ndim.
__global__ void relu_forward_kernel_generic(
    float* __restrict__ output,
    const float* __restrict__ input,
    int numel,
    const int* __restrict__ shape,
    const int* __restrict__ in_strides,
    const int* __restrict__ out_strides,
    int ndim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int linear = idx;
    int in_offset = 0;
    int out_offset = 0;
    int rem = linear;

    for (int d = ndim - 1; d >= 0; --d) {
        int coord = rem % shape[d];
        rem /= shape[d];
        in_offset += coord * in_strides[d];
        out_offset += coord * out_strides[d];
    }

    output[out_offset] = fmaxf(input[in_offset], 0.0f);
}

// AvgPool1d forward kernel: output along last dim
// Supports arbitrary strides for batch and channel dims, last dim stride assumed contiguous? Not necessarily.
// But avg_pool operates on last dimension, and input/output may have arbitrary strides for all dims.
// For each output element (n,c,ow), compute mean over window [ow*stride - padding, ow*stride - padding + kernel_size)
// with count_include_pad=True => divide by kernel_size always.
__global__ void avg_pool1d_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N,
    int C,
    int L_in,
    int L_out,
    int kernel_size,
    int stride,
    int padding,
    const int* __restrict__ in_strides,
    const int* __restrict__ out_strides) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * L_out;
    if (idx >= total) return;

    // Decode idx into n, c, ow
    int ow = idx % L_out;
    int tmp = idx / L_out;
    int c = tmp % C;
    int n = tmp / C;

    // Compute input base offset for (n,c,0)
    int in_base = n * in_strides[0] + c * in_strides[1];
    int out_offset = n * out_strides[0] + c * out_strides[1] + ow * out_strides[2];

    // Start position in input for the window
    int start = ow * stride - padding;
    float sum = 0.0f;

    // Loop over kernel window
    for (int k = 0; k < kernel_size; ++k) {
        int pos = start + k;
        if (pos >= 0 && pos < L_in) {
            int in_offset = in_base + pos * in_strides[2];
            sum += input[in_offset];
        }
        // else contribute 0 for padding
    }

    output[out_offset] = sum / static_cast<float>(kernel_size);
}

// Host launcher functions (extern "C")
extern "C" void relu_forward_launcher(
    float* output,
    const float* input,
    int numel,
    const int* shape,
    const int* in_strides,
    const int* out_strides,
    int ndim,
    cudaStream_t stream) {

    if (numel == 0) return;

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    relu_forward_kernel_generic<<<blocks, threads, 0, stream>>>(
        output, input, numel, shape, in_strides, out_strides, ndim);
}

// For avg_pool1d, we also need to handle arbitrary strides, but we pass explicit strides arrays.
// Since input and output are 3D, we can assume ndim=3, but contract says N,C,L.
// We'll provide a launcher with fixed ndim=3, but pass shape and strides for 3 dims.
extern "C" void avg_pool1d_forward_launcher(
    float* output,
    const float* input,
    int N,
    int C,
    int L_in,
    int L_out,
    int kernel_size,
    int stride,
    int padding,
    const int* in_strides,
    const int* out_strides,
    cudaStream_t stream) {

    if (N == 0 || C == 0 || L_out == 0) return;

    int total = N * C * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    avg_pool1d_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, input, N, C, L_in, L_out, kernel_size, stride, padding,
        in_strides, out_strides);
}
