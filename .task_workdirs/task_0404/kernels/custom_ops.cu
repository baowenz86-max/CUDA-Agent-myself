#include <cuda_runtime.h>

// Fused element-wise abs and threshold kernel.
// Computes output = (abs(input) > threshold) ? abs(input) : 0.0f
// Supports arbitrary strides by operating on linear indices.
template <typename scalar_t>
__global__ void threshold_abs_kernel(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     int64_t total_elements,
                                     scalar_t threshold) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        scalar_t val = input[i];
        scalar_t abs_val = (val < 0) ? -val : val;
        output[i] = (abs_val > threshold) ? abs_val : static_cast<scalar_t>(0.0f);
    }
}

// Launcher for float32 only (contract requires float32).
extern "C" void threshold_abs_forward_launcher(const float* input,
                                                float* output,
                                                int64_t total_elements,
                                                double threshold,
                                                cudaStream_t stream) {
    const int threads = 256;
    const int64_t blocks_64 = (total_elements + threads - 1) / threads;
    const int blocks = static_cast<int>(blocks_64 > 2147483647 ? 2147483647 : blocks_64);
    threshold_abs_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, total_elements, static_cast<float>(threshold));
}
