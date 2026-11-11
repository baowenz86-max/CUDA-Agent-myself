#include <cuda_runtime.h>

// Element-wise absolute value kernel for float32 input.
// Uses float4 vectorized loads for contiguous data and a grid-stride loop.
__global__ void abs_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);
    size_t num_vec = total_elements / 4;

    // Vectorized path
    for (size_t i = idx; i < num_vec; i += stride) {
        float4 val = in4[i];
        val.x = fabsf(val.x);
        val.y = fabsf(val.y);
        val.z = fabsf(val.z);
        val.w = fabsf(val.w);
        out4[i] = val;
    }

    // Tail elements
    for (size_t i = num_vec * 4 + idx; i < total_elements; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Element-wise greater-than-scalar kernel: output = (input > threshold)
// Input is float32, output is bool (stored as unsigned char).
__global__ void gt_scalar_kernel(const float* __restrict__ input,
                                 bool* __restrict__ output,
                                 float threshold,
                                 size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < total_elements; i += stride) {
        output[i] = (input[i] > threshold);
    }
}

// C interface launcher for abs_forward.
// total_elements is number of float elements.
extern "C" void abs_forward_launcher(const float* input,
                                     float* output,
                                     size_t total_elements,
                                     cudaStream_t stream) {
    const int threads = 256;
    // Grid size limited to 65535 in x dimension; use grid-stride loop.
    // For total_elements / 4, cap blocks to something reasonable like 1024*64.
    size_t num_vec = total_elements / 4;
    size_t blocks = (num_vec + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    abs_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

// C interface launcher for gt_scalar_forward.
extern "C" void gt_scalar_forward_launcher(const float* input,
                                           bool* output,
                                           float threshold,
                                           size_t total_elements,
                                           cudaStream_t stream) {
    const int threads = 256;
    size_t blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    gt_scalar_kernel<<<blocks, threads, 0, stream>>>(input, output, threshold, total_elements);
}
