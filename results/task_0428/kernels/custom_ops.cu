#include <cuda_runtime.h>
#include <stdint.h>

// Element-wise abs(x) then divide by integer divisor.
// This kernel is launched with a 1D grid-stride loop over total element count.
// It assumes input is contiguous in memory; binding.cpp is responsible for
// making the input contiguous and allocating the output tensor.
__global__ void abs_div_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t divisor,
                               int64_t total_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Convert divisor to float once. For integer divisor, float division is
    // sufficient and matches torch.div semantics for float32 input.
    float divisor_f = static_cast<float>(divisor);

    for (int64_t i = idx; i < total_elements; i += stride) {
        float val = input[i];
        // Compute absolute value, handling -0.0 correctly (fabsf(-0.0) = 0.0)
        float abs_val = fabsf(val);
        // Divide by divisor. Multiplication by reciprocal could be used, but
        // direct division is clearer and has the same numerical behavior within
        // tolerance. We avoid division by zero if divisor == 0; torch raises an
        // error for integer division by zero, but here divisor may be zero and
        // the expected behavior is not specified. We follow standard float
        // division semantics.
        output[i] = abs_val / divisor_f;
    }
}

extern "C" void abs_div_forward_launcher(const float* input,
                                         float* output,
                                         int64_t divisor,
                                         int64_t total_elements,
                                         cudaStream_t stream) {
    // Launch configuration: use 256 threads per block and enough blocks to
    // cover the data, but avoid launching more than necessary. Grid-stride
    // loop handles overflow.
    const int threads = 256;
    int64_t blocks64 = (total_elements + threads - 1) / threads;
    // Cap blocks to a reasonable maximum (e.g., 65535*2) for grid-stride
    // efficiency, but since we have a grid-stride loop, we can use a fixed
    // maximum grid size. This is especially important for very large tensors.
    // Here total_elements = 2048*256*512 = 268,435,456, so blocks64 = 1,048,576
    // which exceeds the maximum grid dimension in x (2^31-1 for 1D) and also
    // might be suboptimal. We cap at 65535 * 2 to use multiple waves.
    int max_blocks = 65535 * 2;
    int blocks = static_cast<int>(blocks64 > max_blocks ? max_blocks : blocks64);
    
    if (blocks > 0) {
        abs_div_kernel<<<blocks, threads, 0, stream>>>(input, output, divisor, total_elements);
    }
}
