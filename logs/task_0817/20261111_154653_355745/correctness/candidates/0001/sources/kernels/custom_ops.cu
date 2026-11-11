#include <cuda_runtime.h>

// Elementwise cosine kernel
// Each thread processes multiple elements using a grid-stride loop.
// Vectorized (float4) loads/stores if pointers are 16-byte aligned and numel divisible by 4,
// otherwise scalar path.
__global__ void elementwise_cos_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       long numel) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    // Vectorized path: assumes 16-byte alignment and numel % 4 == 0
    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);
    long numel4 = numel / 4;
    long idx4 = idx;
    long stride4 = stride;

    if ((reinterpret_cast<unsigned long long>(input) & 0xF) == 0 &&
        (reinterpret_cast<unsigned long long>(output) & 0xF) == 0 &&
        (numel & 3) == 0) {
        for (long i = idx4; i < numel4; i += stride4) {
            float4 val = in4[i];
            val.x = cosf(val.x);
            val.y = cosf(val.y);
            val.z = cosf(val.z);
            val.w = cosf(val.w);
            out4[i] = val;
        }
    } else {
        // Scalar fallback for misaligned or non-multiple-of-4 lengths
        for (long i = idx; i < numel; i += stride) {
            output[i] = cosf(input[i]);
        }
    }
}

// Elementwise copy (identity) kernel, same vectorization approach
__global__ void elementwise_copy_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        long numel) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);
    long numel4 = numel / 4;
    long idx4 = idx;
    long stride4 = stride;

    if ((reinterpret_cast<unsigned long long>(input) & 0xF) == 0 &&
        (reinterpret_cast<unsigned long long>(output) & 0xF) == 0 &&
        (numel & 3) == 0) {
        for (long i = idx4; i < numel4; i += stride4) {
            out4[i] = in4[i];
        }
    } else {
        for (long i = idx; i < numel; i += stride) {
            output[i] = input[i];
        }
    }
}

// Launcher for cosine
// input: device pointer to input float32 tensor (any stride, but we treat as flat)
// output: device pointer to output float32 tensor (same shape as input)
// numel: total number of elements
// stream: CUDA stream to launch on
extern "C" void custom_cos_launcher(const float* input,
                                   float* output,
                                   long numel,
                                   cudaStream_t stream) {
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    // Cap blocks to avoid launching too many; grid-stride loop handles rest
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    elementwise_cos_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for identity copy
extern "C" void custom_identity_launcher(const float* input,
                                          float* output,
                                          long numel,
                                          cudaStream_t stream) {
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    elementwise_copy_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
