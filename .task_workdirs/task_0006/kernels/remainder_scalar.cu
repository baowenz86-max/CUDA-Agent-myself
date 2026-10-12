#include <cuda_runtime.h>

__global__ void remainder_scalar_kernel(
    const bool* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t divisor,
    long total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long i = idx; i < total_elements; i += stride) {
        int64_t value = (int64_t)input[i];
        // Python-like modulo for non-negative divisor (here 2) but avoid division by zero
        if (divisor == 0) {
            output[i] = 0; // mimic torch? Actually torch.remainder with zero gives NaN for float, but inputs are ints, undefined. We'll set 0.
        } else {
            int64_t rem = value % divisor;
            // Adjust for Python semantics if divisor positive: rem is non-negative
            if (rem < 0) rem += divisor;
            output[i] = rem;
        }
    }
}

extern "C" void remainder_scalar_launcher(
    const bool* input,
    int64_t* output,
    int64_t divisor,
    long total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    remainder_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input, output, divisor, total_elements);
}
