#include <cuda_runtime.h>

// Kernel: applies torch.tril(x, offset) on the last two dimensions.
// All leading dimensions are treated as batch.
// Output must be preallocated with the same shape as input by the binding.
__global__ void tril_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t total_elements,
                            int64_t rows,
                            int64_t cols,
                            int64_t offset) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t col = i % cols;
        int64_t row = (i / cols) % rows;
        if (row >= col - offset) {
            output[i] = input[i];
        } else {
            output[i] = 0.0f;
        }
    }
}

// Host launcher for the tril operation. No PyTorch dependencies.
// The caller (binding .cpp) is responsible for:
//   - allocating output tensor with same shape as input
//   - passing current CUDA stream
//   - checking input is CUDA float32 (or ensuring pointer validity)
extern "C" void tril_forward_launcher(const float* input,
                                      float* output,
                                      int64_t total_elements,
                                      int64_t rows,
                                      int64_t cols,
                                      int64_t offset,
                                      cudaStream_t stream) {
    if (total_elements <= 0) {
        return;  // nothing to write
    }

    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) {
        blocks = 65535;  // max grid dim x; kernel uses grid-stride loop
    }

    tril_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements,
                                                rows, cols, offset);
}
