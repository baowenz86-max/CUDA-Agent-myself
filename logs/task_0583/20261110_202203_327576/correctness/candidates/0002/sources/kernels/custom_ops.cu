// kernels/custom_ops.cu
// CUDA kernels for isinf_like, where_inf_replace, logsoftmax_dim1
// All launchers are extern "C" for binding from C++.

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <math.h>

// Helper: check for inf (either sign)
__device__ __forceinline__ bool is_inf_f32(float x) {
    // IEEE 754: inf has exponent all ones and mantissa zero
    // Use C99 isinf-like bit check
    union { float f; uint32_t u; } conv;
    conv.f = x;
    return (conv.u & 0x7FFFFFFF) == 0x7F800000;
}

// ------------------------- isinf_like -------------------------
// Kernel: elementwise isinf on float input, output bool tensor
__global__ void isinf_like_kernel(
    const float* __restrict__ input,
    bool* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // The model uses contiguous tensors; strides are handled by binding layer
        // by ensuring contiguity or providing data_ptr. Here we assume contiguous,
        // but the launcher can be extended with strides if needed.
        output[idx] = is_inf_f32(input[idx]);
    }
}

extern "C" void isinf_like_launcher(
    const float* input,
    bool* output,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    const int block_size = 256;
    // Grid-stride not needed here because grid is sized to cover numel
    int64_t grid = (numel + block_size - 1) / block_size;
    // Cap grid to avoid overflow (grid can be large but within int range for practical sizes)
    // For numel up to 134M, grid is about 524288, fine.
    isinf_like_kernel<<<(unsigned int)grid, block_size, 0, stream>>>(
        input, output, numel
    );
    // Error check done by binding layer after call (optional)
    // cudaError_t err = cudaGetLastError();
}

// ------------------------- where_inf_replace -------------------------
// Kernel: if mask true, set output to -inf; else copy input
__global__ void where_inf_replace_kernel(
    const bool* __restrict__ mask,
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = mask[idx] ? -INFINITY : input[idx];
    }
}

extern "C" void where_inf_replace_launcher(
    const bool* mask,
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t grid = (numel + block_size - 1) / block_size;
    where_inf_replace_kernel<<<(unsigned int)grid, block_size, 0, stream>>>(
        mask, input, output, numel
    );
}

// ------------------------- logsoftmax_dim1 -------------------------
// Compute log_softmax along dim=1 of a 4D tensor [N, C, H, W]
// The reduction dimension C is contiguous in original layout, but the input
// may be non-contiguous. Binding layer will ensure contiguity or pass strides.
// For simplicity and performance, we assume contiguous input and output, which
// is valid for the given model where x is created by randn and remains contiguous
// through the previous ops (isinf_like and where_inf_replace both output contiguous).
// However, to be robust, we pass strides to launcher. In this kernel, we handle
// arbitrary strides by computing indices. Since the dimension order is fixed as
// [N, C, H, W] and dim=1, we can use the actual strides.
// We'll implement a kernel with one thread per spatial location (H*W) per batch,
// and each thread loops over channels to compute max and sum exp, then writes output.
// This avoids shared memory and handles arbitrary strides elegantly.

// Kernel grid: blockDim.x threads handle consecutive spatial positions (H*W),
// blockIdx.x handles batch index N.
__global__ void logsoftmax_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t strideN,
    int64_t strideC,
    int64_t strideH,
    int64_t strideW
) {
    int64_t hw = H * W;
    int64_t spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spatial_idx >= hw) return;
    int64_t h = spatial_idx / W;
    int64_t w = spatial_idx - h * W;
    // offset to base of this batch's (h,w) location
    int64_t base_offset = blockIdx.y * strideN + h * strideH + w * strideW;
    
    // Find max over channels
    float max_val = -INFINITY;
    for (int64_t c = 0; c < C; ++c) {
        float val = input[base_offset + c * strideC];
        if (val > max_val) max_val = val;
    }
    
    // Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int64_t c = 0; c < C; ++c) {
        float val = input[base_offset + c * strideC];
        sum_exp += __expf(val - max_val);
    }
    
    float log_sum_exp = __logf(sum_exp);
    
    // Write output: input - max - log_sum_exp
    for (int64_t c = 0; c < C; ++c) {
        float val = input[base_offset + c * strideC];
        output[base_offset + c * strideC] = val - max_val - log_sum_exp;
    }
}

extern "C" void logsoftmax_dim1_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t strideN,
    int64_t strideC,
    int64_t strideH,
    int64_t strideW,
    cudaStream_t stream
) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    int64_t hw = H * W;
    const int block_size = 256; // threads per block for spatial dimension
    int64_t grid_x = (hw + block_size - 1) / block_size;
    int64_t grid_y = N; // one block row per batch element
    dim3 grid((unsigned int)grid_x, (unsigned int)grid_y);
    dim3 block(block_size);
    logsoftmax_dim1_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, H, W,
        strideN, strideC, strideH, strideW
    );
}

// Note: The binding layer must ensure input is float32 CUDA tensor, and allocate
// output with same shape and dtype. The strides passed to launcher are the actual
// strides of the tensor, which may not be contiguous. This implementation handles
// arbitrary strides correctly.
