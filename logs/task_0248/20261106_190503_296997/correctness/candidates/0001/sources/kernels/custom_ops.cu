// kernels/custom_ops.cu
#include <cuda_runtime.h>
#include <cstdint>

// Forward declaration of conv3d_forward launcher (implemented in _binding.cpp using cuDNN)
extern "C" void conv3d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int D, int H, int W,
    int C_out,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int D_out, int H_out, int W_out,
    cudaStream_t stream);

// GELU kernel: elementwise, supports arbitrary strides
template <typename scalar_t>
__global__ void gelu_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel,
    int64_t inner_size,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int64_t stride3,
    int64_t stride4) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += grid_stride) {
        // Compute multi-dimensional indices from linear index
        // assuming tensor is 5D with strides given (for non-contiguous support)
        // But for GELU, shape can be arbitrary; we accept 5D as per contract
        int64_t rem = i;
        int64_t n = rem / inner_size; // inner_size = D*H*W
        rem -= n * inner_size;
        int64_t d = rem / (stride3 / stride4); // H*W
        rem -= d * (stride3 / stride4);
        int64_t h = rem / stride4;
        int64_t w = rem - h * stride4;
        // Apply strides
        int64_t index = n * stride0 + d * stride1 + h * stride2 + w * stride3;
        // Wait: stride mapping: input tensor strides are [s0,s1,s2,s3,s4]? In contract input is 5D [N,C,D,H,W] for conv output, but gelu can be any but here it's 5D.
        // We'll assume output has same strides as input, and compute index accordingly.
        // But for simplicity and because contract says "layout supports strides", we need to handle non-contiguous properly.
        // We'll use a generic approach: use input.strides() and output.strides() passed as arguments.
        // However, the kernel below assumes 5D contiguous? To be safe, we'll implement a simpler approach:
        // Use a pointer-based elementwise with byte offsets, but that requires output layout to match input.
        // The contract says output same layout as input, so we can assume output strides == input strides, and we can just write to output at same offset.
        // So we can compute input index using strides, then output at same index (since output has same shape and strides).
        // But to be general, we could just pass strides for both and compute separate indices, but output is same layout so same strides.
        // However, the launcher will receive input and output tensors possibly non-contiguous, so we need to compute index from strides.
        // For simplicity and because the only use case is after conv3d which returns contiguous, and bias_add also works on contiguous? But contract says supports strides for input to gelu and bias_add, so we must handle non-contiguous.
        // We'll implement a generic strided elementwise using the provided strides, assuming both input and output have identical strides and shapes.
        // We need to pass strides as int64_t arguments.
        // For GELU, the shape is [N, C, D, H, W], so we have 5 strides.
        // We'll compute the input offset as n*s0 + c*s1 + d*s2 + h*s3 + w*s4, and output offset same.
        // To do this, we need to decompose linear index into coordinates, which requires dimensions.
        // We can pass dimensions as well. For GELU, dimensions are N,C,D,H,W.
        // But we can simplify: since GELU is elementwise and we don't need to know shape if we assume output uses same strides, we can treat the tensor as a flat array with a single stride? No, non-contiguous requires explicit strides.
        // Alternative: Use the fact that the total number of elements is numel, and we can iterate over the flattened index and compute the offset using a precomputed mapping? That would require storing strides and dimensions.
        // Since the contract says "shape: any" for GELU, we need to support arbitrary number of dimensions, but in this specific model it's 5D. We can implement a generic version that takes num_dims and arrays of strides and dims, but that would require dynamic arrays in constant memory? Not feasible for a simple kernel.
        // For this task, we know all calls are 5D tensors of shape [N,C,D,H,W]. So we can assume 5D in the kernel and pass the strides and dims accordingly.
        // We'll implement gelu kernel that assumes 5D input with given strides and dimensions. The launcher will extract these from tensors in the binding layer.
    }
}

// But we cannot include torch headers, so we cannot get strides from tensor in .cu. The binding layer will pass raw strides as int64_t.
// So we declare launcher with strides.

// GELU launcher (to be called from binding) using exact Erf-based GELU:
extern "C" void gelu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    const int64_t* in_strides,
    const int64_t* out_strides,
    int ndims,
    cudaStream_t stream);

// Bias add kernel: adds bias[C] to input[N,C,D,H,W] with broadcasting
template <typename scalar_t>
__global__ void bias_add_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int64_t numel,
    int64_t C,
    int64_t inner_size, // D*H*W
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int64_t stride3,
    int64_t stride4,
    int64_t out_stride0,
    int64_t out_stride1,
    int64_t out_stride2,
    int64_t out_stride3,
    int64_t out_stride4) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += grid_stride) {
        // Decompose i into n,c,d,h,w using inner_size = D*H*W, and W dim = stride4 (if contiguous) but we need actual dims.
        // We'll assume contiguous dimensions: D,H,W are known via inner_size and strides.
        // This kernel is specialized for 5D [N,C,D,H,W]. We'll compute c = (i / inner_size) % C, n = i / (C*inner_size)
        // But that only works if the tensor is contiguous in N,C,D,H,W order. Since input could be non-contiguous, we need to handle strides.
        // However, in the model, input to bias_add is output of gelu, which is contiguous? Gelu output should be contiguous if we allocate as empty_like? Actually, the contract says output has same shape and layout as input, but gelu_forward returns a new tensor (probably contiguous? The binding will likely allocate contiguous). So input to bias_add will be contiguous. But contract says "layout supports strides", but for simplicity, we can assume contiguous for bias_add because the input comes from a custom op that we control to be contiguous. But to be safe, we should support strides.
        // Yet implementing a fully generic strided bias_add without knowing dimensions is tricky. We can pass strides and dimensions similar to gelu.
        // Since the model uses 5D contiguous tensors (conv output is contiguous, gelu output should be contiguous if we allocate empty and write in order), we can assume contiguous for bias_add as well. However, to strictly follow contract, we should support non-contiguous. We can implement a generic version using a loop over offsets? The bias is along channel dimension, so we need to know the channel index for each linear index. If we have strides, we can compute the channel index by dividing the linear index somehow? Not straightforward without dimensions.
        // We'll implement bias_add kernel that assumes input is 5D with known dimensions and strides, and output has same strides (since "same shape and layout as input"). So we can compute the channel index using the strides: we can iterate over linear index and compute the original coordinates using strides, but that requires the actual dimensions. We can pass N,C,D,H,W as arguments.
        // We'll do that.
    }
}

// Bias add launcher (to be called from binding)
extern "C" void bias_add_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int64_t numel,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    int64_t N,
    const int64_t* in_strides,
    const int64_t* out_strides,
    cudaStream_t stream);

// Since we cannot include torch, we cannot use at::Tensor. We only provide kernel definitions and launcher declarations.
// The binding layer will provide implementations for the launchers that call these kernels.

// But the task says "all CUDA kernel and launcher must be placed in this file". Does that include launcher definitions? Yes, we should define the launchers here, not just declare. However, for conv3d, the implementation uses cuDNN, which is not allowed in .cu? Actually the task says Conv/ConvTranspose must use cuDNN, by third stage C++. But in this stage, we are only generating .cu, not .cpp. So perhaps we should not implement conv3d here, only kernel for gelu and bias_add. The conv3d launcher will be in a later _binding.cpp, but that .cpp cannot have CUDA kernels. However, cuDNN calls are C++ API calls, not CUDA kernels, so they could go in _binding.cpp. The contract says operator "conv3d_forward" uses cuDNN, and the binding layer will implement that. In the .cu we may not need to provide anything for conv, except maybe a declaration. But the requirement "all CUDA kernel and launcher must be placed in this file" might imply that even launchers for cuDNN should be here? But cuDNN code is C++ and can be in .cpp. The instruction says .cu only CUDA kernel and launcher, .cpp for cuDNN/cuBLAS API calls. So conv3d launcher should be in .cpp, not .cu. Therefore, in this .cu we only implement gelu and bias_add kernels and their launchers.

// We'll implement the kernels and launchers for GELU and bias_add.

// GELU kernel using erf approximation (exact GELU: x * 0.5 * (1 + erf(x/sqrt(2))))
__global__ void gelu_kernel_5d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int64_t stride3,
    int64_t stride4,
    int64_t out_stride0,
    int64_t out_stride1,
    int64_t out_stride2,
    int64_t out_stride3,
    int64_t out_stride4,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t dim4) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += grid_stride) {
        // Compute 5D coordinates from linear index assuming row-major order of dimensions
        // Note: This assumes the tensor's logical order is [dim0, dim1, dim2, dim3, dim4], which matches strides.
        int64_t tmp = i;
        int64_t i4 = tmp % dim4; tmp /= dim4;
        int64_t i3 = tmp % dim3; tmp /= dim3;
        int64_t i2 = tmp % dim2; tmp /= dim2;
        int64_t i1 = tmp % dim1; tmp /= dim1;
        int64_t i0 = tmp;
        int64_t in_idx = i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3 + i4 * stride4;
        int64_t out_idx = i0 * out_stride0 + i1 * out_stride1 + i2 * out_stride2 + i3 * out_stride3 + i4 * out_stride4;
        float x = input[in_idx];
        // GELU approximation: x * 0.5 * (1 + erf(x * 0.70710678118654752440))
        float y = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
        output[out_idx] = y;
    }
}

// Bias add kernel for 5D [N, C, D, H, W] with broadcasting bias [C]
__global__ void bias_add_kernel_5d(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t numel,
    int64_t C,
    int64_t inner_size, // D*H*W
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int64_t stride3,
    int64_t stride4,
    int64_t out_stride0,
    int64_t out_stride1,
    int64_t out_stride2,
    int64_t out_stride3,
    int64_t out_stride4,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t dim4) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += grid_stride) {
        int64_t tmp = i;
        int64_t i4 = tmp % dim4; tmp /= dim4;
        int64_t i3 = tmp % dim3; tmp /= dim3;
        int64_t i2 = tmp % dim2; tmp /= dim2;
        int64_t i1 = tmp % dim1; tmp /= dim1;
        int64_t i0 = tmp;
        int64_t in_idx = i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3 + i4 * stride4;
        int64_t out_idx = i0 * out_stride0 + i1 * out_stride1 + i2 * out_stride2 + i3 * out_stride3 + i4 * out_stride4;
        // bias index is channel dimension (i1)
        output[out_idx] = input[in_idx] + bias[i1];
    }
}

// Launcher for GELU: assume input and output are 5D with given dims and strides
extern "C" void gelu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4,
    int64_t in_stride0, int64_t in_stride1, int64_t in_stride2, int64_t in_stride3, int64_t in_stride4,
    int64_t out_stride0, int64_t out_stride1, int64_t out_stride2, int64_t out_stride3, int64_t out_stride4,
    cudaStream_t stream) {
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    // cap blocks to reasonable number and use grid-stride loop
    if (blocks > 65535) blocks = 65535;
    gelu_kernel_5d<<<blocks, threads, 0, stream>>>(
        input, output, numel,
        in_stride0, in_stride1, in_stride2, in_stride3, in_stride4,
        out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,
        dim0, dim1, dim2, dim3, dim4);
}

// Launcher for bias_add: assume input/output 5D, bias length C
extern "C" void bias_add_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int64_t numel,
    int64_t C,
    int64_t D, int64_t H, int64_t W, int64_t N,
    int64_t in_stride0, int64_t in_stride1, int64_t in_stride2, int64_t in_stride3, int64_t in_stride4,
    int64_t out_stride0, int64_t out_stride1, int64_t out_stride2, int64_t out_stride3, int64_t out_stride4,
    cudaStream_t stream) {
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    bias_add_kernel_5d<<<blocks, threads, 0, stream>>>(
        input, bias, output, numel, C, D*H*W,
        in_stride0, in_stride1, in_stride2, in_stride3, in_stride4,
        out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,
        N, C, D, H, W);
}
