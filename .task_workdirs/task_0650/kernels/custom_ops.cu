#include <cuda_runtime.h>
#include <cstdint>

// Compute isnan for each element, output bool (as uint8_t in memory)
__global__ void isnan_kernel(const float* __restrict__ input, uint8_t* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = (input[i] != input[i]) ? 1 : 0;
    }
}

// Gather along dimension dim (int64_t indices)
__global__ void gather_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ index,
    float* __restrict__ output,
    int64_t outer_size,   // product of dims before dim
    int64_t dim_size,     // size of gather dimension
    int64_t inner_size,   // product of dims after dim
    int64_t total_output  // outer_size * inner_size (index shape numel)
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_output; i += stride) {
        int64_t inner = i % inner_size;
        int64_t outer = i / inner_size;
        int64_t index_val = index[i];
        // torch.gather uses wrap-around for out-of-bounds indices, but contract says indices are valid
        // For safety, clamp to [0, dim_size-1] in debug? No, torch.gather allows negative indices with wrap.
        // Here we follow torch semantics: if idx < 0, idx += dim_size; if still out of range, undefined? Actually torch.gather requires in range.
        // Index provided by torch.randint(0, dim_size) so always valid.
        if (index_val < 0) index_val += dim_size; // not needed but safe
        if (index_val >= dim_size) index_val %= dim_size; // not needed
        int64_t input_offset = outer * dim_size * inner_size + index_val * inner_size + inner;
        output[i] = input[input_offset];
    }
}

extern "C" void isnan_forward_launcher(const float* input, uint8_t* output, int64_t numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0 && numel > 0) blocks = 1;
    if (numel == 0) return;
    isnan_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void gather_forward_launcher(
    const float* input,
    const int64_t* index,
    float* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    cudaStream_t stream
) {
    int64_t total_output = outer_size * inner_size;
    if (total_output == 0) return;
    int threads = 256;
    int blocks = (int)((total_output + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    gather_kernel<<<blocks, threads, 0, stream>>>(input, index, output, outer_size, dim_size, inner_size, total_output);
}
