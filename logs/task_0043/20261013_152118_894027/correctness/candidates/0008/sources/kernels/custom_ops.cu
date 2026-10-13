#include <cuda_runtime.h>
#include <cstdint>

// Helper to compute index from multi-dimensional strides
template <typename T>
__device__ __forceinline__ T get_strided_index(const T* strides, const T* shape, int ndim, const T* indices) {
    T idx = 0;
    for (int i = 0; i < ndim; ++i) {
        idx += indices[i] * strides[i];
    }
    return idx;
}

// Kernel for view_as_real: complex64 (float2) input to float32 output with last dim size 2.
// Input: complex64 tensor with shape [B, C, H, W] (or any dimensions) on CPU. We copy to CUDA memory.
// Output: float32 tensor with shape [..., 2].
// The launcher will allocate output on CUDA and call this kernel.
// We use a 1D grid: each thread handles one element of the original complex tensor (all except last dim).
// For each complex element at flat index i, we write output[2*i] = real, output[2*i+1] = imag.
// This assumes input is contiguous? But contract says supports strides. We need to handle general strides.
// However, the input is on CPU, and we need to copy data to GPU. The binding will likely use a contiguous copy or pass raw pointer with strides.
// For simplicity and performance, we assume the input is contiguous, but we can handle strides by using a kernel that iterates over logical linear index and computes multi-dimensional indices.
// Since input is complex64 on CPU, we must first copy to GPU. The binding will handle that; launcher expects a GPU pointer for complex input.
// We'll create a kernel that reads from GPU complex data with strides and writes to GPU output contiguous.
__global__ void view_as_real_kernel(const float2* __restrict__ input,
                                    float* __restrict__ output,
                                    const int64_t total_complex_elements,
                                    const int64_t* __restrict__ input_strides,
                                    const int64_t* __restrict__ input_shape,
                                    int ndim) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_complex_elements; i += stride) {
        // Compute multi-dimensional indices for this linear index over the non-last dimensions.
        int64_t remainder = i;
        int64_t flat_input_idx = 0;
        // ndim is the number of dimensions of the original complex tensor (e.g., B, C, H, W)
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t dim_size = input_shape[d];
            int64_t coord = remainder % dim_size;
            remainder /= dim_size;
            flat_input_idx += coord * input_strides[d];
        }
        float2 val = input[flat_input_idx];
        output[2 * i] = val.x;
        output[2 * i + 1] = val.y;
    }
}

// Kernel for einsum bchw->bhwc on a 5D tensor where last dim size is 2.
// Input shape: [B, C, H, W, 2], output shape: [B, H, W, C, 2].
// We treat the last dimension (size 2) as a unit and copy both elements.
// We'll use a 1D grid over total number of output elements excluding the last dim? Actually output total = B*H*W*C*2.
// We can have each thread handle one output element (flat index).
// General strides supported via input and output strides.
__global__ void einsum_permute_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const int64_t total_output_elements,
                                      const int64_t* __restrict__ input_strides, // 5 strides
                                      const int64_t* __restrict__ output_strides, // 5 strides
                                      const int64_t* __restrict__ output_shape, // 5 dims
                                      int ndim) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_output_elements; i += stride) {
        // Compute output multi-dimensional index from flat i.
        int64_t remainder = i;
        int64_t input_idx = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t coord = remainder % output_shape[d];
            remainder /= output_shape[d];
            // Map output coord to input coord: b->b, h->c, w->h, c->w, 2->2
            int input_dim;
            if (d == 0) input_dim = 0; // b
            else if (d == 1) input_dim = 2; // output h -> input h (since input order: b,c,h,w)
            else if (d == 2) input_dim = 3; // output w -> input w
            else if (d == 3) input_dim = 1; // output c -> input c
            else input_dim = 4; // last 2
            input_idx += coord * input_strides[input_dim];
        }
        output[i] = input[input_idx];
    }
}

// Kernel for clip
__global__ void clip_kernel(float* __restrict__ data,
                            int64_t total_elements,
                            float min_val,
                            float max_val) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        float val = data[i];
        val = fminf(fmaxf(val, min_val), max_val);
        data[i] = val;
    }
}

// Launchers (extern "C" for binding)
extern "C" void view_as_real_launcher(const float2* input, float* output,
                                      int64_t total_complex_elements,
                                      const int64_t* input_strides,
                                      const int64_t* input_shape,
                                      int ndim,
                                      cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (total_complex_elements + block_size - 1) / block_size;
    if (grid_size > 2147483647) grid_size = 2147483647;
    view_as_real_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, output, total_complex_elements, input_strides, input_shape, ndim);
}

extern "C" void einsum_permute_launcher(const float* input, float* output,
                                        int64_t total_output_elements,
                                        const int64_t* input_strides,
                                        const int64_t* output_strides,
                                        const int64_t* output_shape,
                                        int ndim,
                                        cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (total_output_elements + block_size - 1) / block_size;
    if (grid_size > 2147483647) grid_size = 2147483647;
    einsum_permute_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, output, total_output_elements, input_strides, output_strides, output_shape, ndim);
}

extern "C" void clip_launcher(float* data, int64_t total_elements,
                              float min_val, float max_val,
                              cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 2147483647) grid_size = 2147483647;
    clip_kernel<<<(int)grid_size, block_size, 0, stream>>>(data, total_elements, min_val, max_val);
}
