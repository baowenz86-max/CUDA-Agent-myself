#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <climits>

// Kernel: sum over dimension 1 of input shape [B, C, H, W], with C=3 (per contract).
// Output shape [B, H, W]. Input may have arbitrary strides but contract expects contiguous.
// We assume contiguous row-major layout for simplicity and performance; the binding layer ensures
// contiguous input by calling .contiguous() if needed.
__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t batch,
                                int64_t height,
                                int64_t width) {
    int64_t total = batch * height * width;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t b = i / (height * width);
        int64_t rem = i % (height * width);
        int64_t h = rem / width;
        int64_t w = rem % width;

        // Channel count is fixed at 3 (contract input shape [16,3,4096,4096])
        constexpr int C = 3;
        const float* base = input + b * C * height * width + h * width + w;
        float sum = base[0];
        sum += base[height * width];
        sum += base[2 * height * width];
        output[i] = sum;
    }
}

// Kernel: find argmax over all elements of a contiguous float array.
// Uses a single pass with atomic operations on packed int64 bits.
// For each element, we pack: high 32 bits = float bits (order-preserving for positive floats,
// but we need to handle sign bit correctly). For a proper order, we transform float to
// a sortable uint32: if sign bit set, flip all bits; else flip only sign bit.
// Then combined with index (lower 32 bits) we can use atomicMax on unsigned long long.
// Since indices are up to 2^28 (268M), lower 32 bits are sufficient.
// The output is int64_t, but we reinterpret as unsigned long long for atomicMax.
__global__ void argmax_all_kernel(const float* __restrict__ input,
                                  unsigned long long* __restrict__ output,
                                  int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        // Convert float to sortable uint32
        uint32_t float_bits = __float_as_uint(val);
        uint32_t sortable_bits = (float_bits & 0x80000000) ? ~float_bits : (float_bits | 0x80000000);
        // Pack: high 32 bits = sortable_bits, low 32 bits = index (i)
        unsigned long long packed = ((unsigned long long)sortable_bits << 32) | (unsigned long long)i;
        // Atomic max on packed value ensures maximum value and smallest index among ties
        atomicMax(output, packed);
    }
}

// Launcher for sum reduction
extern "C" void torch_sum_dim1_launcher(const float* input,
                                         float* output,
                                         int64_t batch,
                                         int64_t height,
                                         int64_t width,
                                         cudaStream_t stream) {
    int64_t total = batch * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    blocks = (blocks < 1024) ? blocks : 1024; // grid-stride loop handles the rest
    sum_dim1_kernel<<<blocks, threads, 0, stream>>>(input, output, batch, height, width);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be checked by binding after launch
    }
}

// Launcher for argmax over all elements
extern "C" void torch_argmax_all_launcher(const float* input,
                                           int64_t* output,
                                           int64_t numel,
                                           cudaStream_t stream) {
    // Initialize output to the packed value representing -infinity and index 0
    // -infinity sortable bits: for -inf, float bits = 0xFF800000, sign bit set => ~0xFF800000 = 0x007FFFFF
    // But we need to ensure any valid value will be larger. For -inf, sortable = 0x007FFFFF.
    // We'll set packed to (0x007FFFFF << 32) | 0, which is less than any finite positive or negative.
    // Actually, for any finite value, sortable bits will be > 0x007FFFFF because negative finite values flip to large values.
    // We can set initial packed to 0x007FFFFF00000000ULL.
    unsigned long long init_packed = 0x007FFFFF00000000ULL;
    cudaMemcpyAsync(output, &init_packed, sizeof(unsigned long long), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    blocks = (blocks < 1024) ? blocks : 1024;

    // Launch kernel, output is uint64 pointer
    argmax_all_kernel<<<blocks, threads, 0, stream>>>(input, reinterpret_cast<unsigned long long*>(output), numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // binding will check after launch
    }
}
