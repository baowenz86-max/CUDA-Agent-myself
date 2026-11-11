#include <cuda_runtime.h>

// Kernel for mean along a dimension, keepdim=True
// x: input data (contiguous in memory, but we handle arbitrary strides outside or assume contiguous after binding)
// output: output data with reduced dimension size 1
// total_output_elements: number of elements in output
// dim_size: size of the dimension being reduced
// inner_size: product of dimensions after dim
// outer_size: product of dimensions before dim
__global__ void mean_dim_kernel(const float* __restrict__ x,
                                float* __restrict__ output,
                                int total_output_elements,
                                int dim_size,
                                int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_output_elements; i += stride) {
        // output linear index i -> output indices (outer, 0, inner)
        int inner = i % inner_size;
        int outer = i / inner_size;
        // For keepdim=True, output shape has dim dimension of length 1, so same outer and inner mapping.
        // Input index: (outer, d, inner) where d in [0, dim_size)
        float sum = 0.0f;
        int base_input_index = outer * dim_size * inner_size + inner;
        for (int d = 0; d < dim_size; ++d) {
            sum += x[base_input_index + d * inner_size];
        }
        output[i] = sum / (float)dim_size;
    }
}

// Kernel for concatenation along a dimension
// out: output tensor
// in1: first input tensor (mean_x), in2: second input tensor (x)
// total_output_elements: number of elements in output
// dim_size1: size of dim in in1
// dim_size2: size of dim in in2
// inner_size: product of dimensions after dim
// outer_size: product of dimensions before dim
__global__ void cat_dim_kernel(const float* __restrict__ in1,
                               const float* __restrict__ in2,
                               float* __restrict__ out,
                               int total_output_elements,
                               int dim_size1,
                               int dim_size2,
                               int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int dim_total = dim_size1 + dim_size2;
    for (int i = idx; i < total_output_elements; i += stride) {
        // decompose output index into (outer, d, inner)
        int inner = i % inner_size;
        int tmp = i / inner_size;
        int d = tmp % dim_total;
        int outer = tmp / dim_total;
        float val;
        if (d < dim_size1) {
            // from in1
            int in1_index = outer * dim_size1 * inner_size + d * inner_size + inner;
            val = in1[in1_index];
        } else {
            // from in2
            int d2 = d - dim_size1;
            int in2_index = outer * dim_size2 * inner_size + d2 * inner_size + inner;
            val = in2[in2_index];
        }
        out[i] = val;
    }
}

// Launcher for mean_dim
// x: pointer to input data (assumed contiguous float32)
// output: pointer to output data (allocated by binding)
// outer_size: product of dimensions before reduce dim
// dim_size: reduction dimension size
// inner_size: product of dimensions after reduce dim
// stream: CUDA stream
extern "C" void mean_dim_launcher(const float* x,
                                  float* output,
                                  int outer_size,
                                  int dim_size,
                                  int inner_size,
                                  cudaStream_t stream) {
    int total_output_elements = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;
    // Limit grid size to avoid launching too many blocks; use grid-stride loop
    if (blocks > 65535) blocks = 65535;
    mean_dim_kernel<<<blocks, threads, 0, stream>>>(
        x, output, total_output_elements, dim_size, inner_size);
}

// Launcher for cat_dim
// in1: pointer to mean_x (contiguous)
// in2: pointer to x (contiguous)
// out: output pointer
// outer_size: product of dimensions before cat dim
// dim_size1: size of cat dim in in1
// dim_size2: size of cat dim in in2
// inner_size: product of dimensions after cat dim
// stream: CUDA stream
extern "C" void cat_dim_launcher(const float* in1,
                                  const float* in2,
                                  float* out,
                                  int outer_size,
                                  int dim_size1,
                                  int dim_size2,
                                  int inner_size,
                                  cudaStream_t stream) {
    int total_output_elements = outer_size * (dim_size1 + dim_size2) * inner_size;
    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cat_dim_kernel<<<blocks, threads, 0, stream>>>(
        in1, in2, out, total_output_elements,
        dim_size1, dim_size2, inner_size);
}
