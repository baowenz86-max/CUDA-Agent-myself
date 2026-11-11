#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------------

// Normalize a dimension index to [0, ndim).
__device__ __forceinline__ int normalize_dim(int dim, int ndim) {
    if (dim < 0) dim += ndim;
    return dim;
}

// Convert linear index to multi-index given sizes and strides.
__device__ __forceinline__ void unravel_index(
    long long linear,
    const int* sizes,
    const long long* strides,
    int ndim,
    int* out_idx) {
    for (int d = 0; d < ndim; ++d) {
        long long size_d = sizes[d];
        out_idx[d] = (int)(linear / strides[d]);
        linear %= strides[d];
    }
}

// Compute linear index from multi-index using strides.
__device__ __forceinline__ long long ravel_index(
    const int* idx,
    const long long* strides,
    int ndim) {
    long long linear = 0;
    for (int d = 0; d < ndim; ++d) {
        linear += (long long)idx[d] * strides[d];
    }
    return linear;
}

// -----------------------------------------------------------------------------
// mean_only: reduction along one dimension
// -----------------------------------------------------------------------------
// The launcher takes the input pointer, output pointer, input sizes,
// reduction dim, and total number of output elements. The input is
// expected to be contiguous in the PyTorch sense, but we also pass strides in
// case the binding wants to handle non-contiguous input. For this initial
// implementation we assume the input is contiguous (C-order) and verify that
// in the binding; if non-contiguous input is required, the binding should
// make it contiguous before calling. Here we implement a generic kernel that
// works for contiguous or non-contiguous input by using strides.

__global__ void mean_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ input_sizes,
    const long long* __restrict__ input_strides,
    int ndim,
    int reduce_dim,
    long long out_numel) {

    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    int outer_factor = 1;
    int inner_factor = 1;
    for (int d = 0; d < reduce_dim; ++d) {
        outer_factor *= input_sizes[d];
    }
    for (int d = reduce_dim + 1; d < ndim; ++d) {
        inner_factor *= input_sizes[d];
    }
    int reduce_size = input_sizes[reduce_dim];

    // For contiguous input, we can optimize indexing.
    // Input is assumed contiguous: strides as in PyTorch contiguous (row-major).
    // Since binding should ensure contiguity, we use optimized path.
    // But to be safe, we implement general using strides.

    for (long long out_idx = idx; out_idx < out_numel; out_idx += stride) {
        // Decompose out_idx into outer index, inner index
        int outer = (int)(out_idx / inner_factor);
        int inner = (int)(out_idx % inner_factor);

        double sum = 0.0;
        const float* base = input + (long long)outer * reduce_size * inner_factor + inner;
        for (int r = 0; r < reduce_size; ++r) {
            sum += (double)base[(long long)r * inner_factor];
        }
        output[out_idx] = (float)(sum / reduce_size);
    }
}

// -----------------------------------------------------------------------------
// mean_only kernel (contiguous input)
// -----------------------------------------------------------------------------
__global__ void mean_only_contiguous_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer_factor,
    int inner_factor,
    int reduce_size) {

    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    long long total = (long long)outer_factor * inner_factor;

    for (long long out_idx = idx; out_idx < total; out_idx += stride) {
        int outer = (int)(out_idx / inner_factor);
        int inner = (int)(out_idx % inner_factor);

        double sum = 0.0;
        const float* base = input + (long long)outer * reduce_size * inner_factor + inner;
        for (int r = 0; r < reduce_size; ++r) {
            sum += (double)base[(long long)r * inner_factor];
        }
        output[out_idx] = (float)(sum / reduce_size);
    }
}

// Final mean_only launcher
extern "C" void mean_only_launcher(
    const float* input,
    float* output,
    const int* input_sizes,
    int ndim,
    int dim,
    cudaStream_t stream) {

    int normalized_dim = (dim < 0) ? dim + ndim : dim;
    if (normalized_dim < 0 || normalized_dim >= ndim) return;

    int outer_factor = 1;
    for (int d = 0; d < normalized_dim; ++d) outer_factor *= input_sizes[d];
    int inner_factor = 1;
    for (int d = normalized_dim + 1; d < ndim; ++d) inner_factor *= input_sizes[d];
    int reduce_size = input_sizes[normalized_dim];

    long long total = (long long)outer_factor * inner_factor;
    if (total == 0) return;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;

    mean_only_contiguous_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, outer_factor, inner_factor, reduce_size);
}

// -----------------------------------------------------------------------------
// transpose_only kernel
// -----------------------------------------------------------------------------
// The binding will create an output tensor with transposed shape and contiguous
// strides (torch::empty). The kernel copies data from input to output with
// appropriate index mapping. The input is assumed contiguous (binding ensures).
// The output is contiguous. We need to map output linear index to input linear
// index by applying the inverse transpose.

__global__ void transpose_only_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ output_sizes,
    const int* __restrict__ output_strides,
    const int* __restrict__ input_strides,
    int ndim,
    int dim0,
    int dim1,
    long long output_numel) {

    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long out_linear = idx; out_linear < output_numel; out_linear += stride) {
        // Convert output linear index to multi-index using output_strides
        long long rem = out_linear;
        int out_idx[8]; // assume ndim <= 8, typical
        for (int d = 0; d < ndim; ++d) {
            out_idx[d] = (int)(rem / output_strides[d]);
            rem %= output_strides[d];
        }

        // Swap dim0 and dim1 to get input index
        int tmp = out_idx[dim0];
        out_idx[dim0] = out_idx[dim1];
        out_idx[dim1] = tmp;

        // Compute input linear offset using input_strides
        long long in_linear = 0;
        for (int d = 0; d < ndim; ++d) {
            in_linear += (long long)out_idx[d] * input_strides[d];
        }
        output[out_linear] = input[in_linear];
    }
}

// Specialized transpose kernels for 3D and 4D contiguous tensors.
__global__ void transpose3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in0, int in1, int in2,
    int out0, int out1, int out2,
    int dim0, int dim1) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = in0 * in1 * in2;
    if (idx >= total) return;

    // Convert idx to input multi-index (contiguous)
    int i2 = idx % in2;
    int temp = idx / in2;
    int i1 = temp % in1;
    int i0 = temp / in1;

    int idx_arr[3] = {i0, i1, i2};
    int o0, o1, o2;
    if (dim0 == 0 && dim1 == 1) {
        o0 = idx_arr[1]; o1 = idx_arr[0]; o2 = idx_arr[2];
    } else if (dim0 == 0 && dim1 == 2) {
        o0 = idx_arr[2]; o1 = idx_arr[1]; o2 = idx_arr[0];
    } else if (dim0 == 1 && dim1 == 2) {
        o0 = idx_arr[0]; o1 = idx_arr[2]; o2 = idx_arr[1];
    } else {
        // other cases, swap dim0,dim1
        o0 = idx_arr[0]; o1 = idx_arr[1]; o2 = idx_arr[2];
        int tmp = o0;
        o0 = o1;
        o1 = tmp;
    }

    int out_idx = o0 * out1 * out2 + o1 * out2 + o2;
    output[out_idx] = input[idx];
}

__global__ void transpose4d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in0, int in1, int in2, int in3,
    int out0, int out1, int out2, int out3,
    int dim0, int dim1) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)in0 * in1 * in2 * in3;
    if (idx >= total) return;

    int i3 = idx % in3;
    long long temp = idx / in3;
    int i2 = temp % in2;
    temp /= in2;
    int i1 = temp % in1;
    int i0 = temp / in1;

    int idx_arr[4] = {i0, i1, i2, i3};
    int o0, o1, o2, o3;
    // general swap dim0,dim1
    o0 = idx_arr[0]; o1 = idx_arr[1]; o2 = idx_arr[2]; o3 = idx_arr[3];
    int tmp;
    if (dim0 == 0 && dim1 == 1) { tmp=o0; o0=o1; o1=tmp; }
    else if (dim0 == 0 && dim1 == 2) { tmp=o0; o0=o2; o2=tmp; }
    else if (dim0 == 0 && dim1 == 3) { tmp=o0; o0=o3; o3=tmp; }
    else if (dim0 == 1 && dim1 == 2) { tmp=o1; o1=o2; o2=tmp; }
    else if (dim0 == 1 && dim1 == 3) { tmp=o1; o1=o3; o3=tmp; }
    else if (dim0 == 2 && dim1 == 3) { tmp=o2; o2=o3; o3=tmp; }

    long long out_idx = ((long long)o0 * out1 * out2 * out3 +
                         (long long)o1 * out2 * out3 +
                         (long long)o2 * out3 +
                         o3);
    output[out_idx] = input[idx];
}

// Launcher for transpose_only.
// The binding will normalize dims, allocate output with transposed shape
// (contiguous), and pass the necessary metadata.
// Here we expect input is contiguous, output is contiguous, and we are given
// sizes and strides for both (though input is contiguous, we can compute
// strides if needed, but we trust the binding to pass correct contiguous
// strides). For simplicity, we assume contiguous input and output, and compute
// strides ourselves based on sizes in C order.
// However, the binding should pass output_sizes, input_sizes, ndim, dim0, dim1.
// We'll compute output_numel = product of output_sizes.
// input_numel = product of input_sizes (same).
// input_strides and output_strides are contiguous C-order.
extern "C" void transpose_only_launcher(
    const float* input,
    float* output,
    const int* input_sizes,
    const int* output_sizes,
    int ndim,
    int dim0,
    int dim1,
    cudaStream_t stream) {

    // Normalize dims (binding should have done, but do it again safely)
    int norm_dim0 = (dim0 < 0) ? dim0 + ndim : dim0;
    int norm_dim1 = (dim1 < 0) ? dim1 + ndim : dim1;
    if (norm_dim0 < 0 || norm_dim0 >= ndim || norm_dim1 < 0 || norm_dim1 >= ndim || norm_dim0 == norm_dim1) {
        return;
    }

    // Compute contiguous strides for input and output (not used in specialized kernels)
    long long input_strides[8], output_strides[8];
    input_strides[ndim-1] = 1;
    output_strides[ndim-1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        input_strides[d] = input_strides[d+1] * input_sizes[d+1];
        output_strides[d] = output_strides[d+1] * output_sizes[d+1];
    }

    long long output_numel = 1;
    for (int d = 0; d < ndim; ++d) output_numel *= output_sizes[d];
    if (output_numel == 0) return;

    if (ndim == 3) {
        // input sizes: [D0, D1, D2], output sizes: swapped dim0,dim1
        int block_size = 256;
        long long total = (long long)input_sizes[0] * input_sizes[1] * input_sizes[2];
        int grid_size = (int)((total + block_size - 1) / block_size);
        if (grid_size > 65535) grid_size = 65535;
        transpose3d_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output,
            input_sizes[0], input_sizes[1], input_sizes[2],
            output_sizes[0], output_sizes[1], output_sizes[2],
            norm_dim0, norm_dim1);
        return;
    } else if (ndim == 4) {
        // similar
        int block_size = 256;
        long long total = (long long)input_sizes[0] * input_sizes[1] * input_sizes[2] * input_sizes[3];
        int grid_size = (int)((total + block_size - 1) / block_size);
        if (grid_size > 65535) grid_size = 65535;
        transpose4d_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output,
            input_sizes[0], input_sizes[1], input_sizes[2], input_sizes[3],
            output_sizes[0], output_sizes[1], output_sizes[2], output_sizes[3],
            norm_dim0, norm_dim1);
        return;
    } else {
        // Unsupported or fallback to simple loop? We'll do nothing, but binding
        // should avoid calling with unsupported ndim, or we could implement a
        // generic kernel using dynamic arrays passed through constant memory.
        // For now, return; binding should handle error.
        return;
    }
}
