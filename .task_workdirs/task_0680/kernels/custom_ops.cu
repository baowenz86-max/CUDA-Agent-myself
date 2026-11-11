// custom_ops.cu
#include <cuda_runtime.h>
#include <cstdint>

// Kernel for concatenating two tensors along a given dimension.
// Assumes dim is 1 for this task, but general implementation for any dim is provided.
// Inputs are contiguous float32 tensors.
__global__ void custom_cat_kernel(
    float* __restrict__ output,
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    int64_t total_output_size,
    int64_t dim,
    int64_t size1, int64_t size2,
    int64_t stride1, int64_t stride2,
    int64_t outer_size,
    int64_t inner_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_output_size; i += stride) {
        // Decompose output index into (outer, channel, inner)
        int64_t inner = i % inner_size;
        int64_t tmp = i / inner_size;
        int64_t channel = tmp % (size1 + size2);
        int64_t outer = tmp / (size1 + size2);

        float val;
        if (channel < size1) {
            int64_t input_idx = outer * (stride1 * size1) + channel * stride1 + inner;
            val = input1[input_idx];
        } else {
            int64_t channel2 = channel - size1;
            int64_t input_idx = outer * (stride2 * size2) + channel2 * stride2 + inner;
            val = input2[input_idx];
        }
        output[i] = val;
    }
}

// Kernel for nansum along a given dimension (dim=1 in this task).
// Each thread handles one output element (batch, spatial).
// Input is assumed contiguous float32 with shape [outer_size, reduction_size, inner_size].
__global__ void custom_nansum_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t outer_size,
    int64_t reduction_size,
    int64_t inner_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    if (idx >= total) return;

    int64_t inner = idx % inner_size;
    int64_t outer = idx / inner_size;

    float sum = 0.0f;
    const float* base = input + outer * reduction_size * inner_size + inner;
    for (int64_t c = 0; c < reduction_size; ++c) {
        float val = base[c * inner_size];
        if (!isnan(val)) {
            sum += val;
        }
    }
    output[outer * inner_size + inner] = sum;
}

// Host launcher for custom_cat
extern "C" void custom_cat_launcher(
    float* output,
    const float* input1,
    const float* input2,
    int64_t dim,
    int64_t size1,
    int64_t size2,
    int64_t outer_size,
    int64_t inner_size,
    int64_t total_output_size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (int)((total_output_size + block_size - 1) / block_size);
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    custom_cat_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input1, input2, total_output_size, dim,
        size1, size2, inner_size, inner_size,  // stride1 = stride2 = inner_size for contiguous
        outer_size, inner_size
    );
}

// Host launcher for custom_nansum
extern "C" void custom_nansum_launcher(
    float* output,
    const float* input,
    int64_t outer_size,
    int64_t reduction_size,
    int64_t inner_size,
    cudaStream_t stream
) {
    int64_t total = outer_size * inner_size;
    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    custom_nansum_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, outer_size, reduction_size, inner_size
    );
}
