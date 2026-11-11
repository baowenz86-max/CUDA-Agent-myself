#include <cuda_runtime.h>
#include <math.h>

// Element-wise softplus: y = x for x > 20, else log1p(exp(x))
__global__ void softplus_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                long long total_elements) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        float x = input[i];
        if (x > 20.0f) {
            output[i] = x;
        } else {
            output[i] = log1pf(expf(x));
        }
    }
}

// Sum over dimension 1 (size C) with keepdim=true.
// Input shape [N, C, H, W], output shape [N, 1, H, W].
// Input is contiguous in H,W and C, but stride in N is C*H*W = C*HW.
// Output stride: [C*HW, HW, W, 1] where HW = H*W, but output dim1 size is 1.
// We compute output[b,0,h,w] = sum_{c=0}^{C-1} input[b,c,h,w]
__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int N, int C, int H, int W) {
    long long HW = (long long)H * W;
    long long total_output = (long long)N * HW; // number of output elements
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total_output; i += stride) {
        // i corresponds to output index: b * HW + (h*W + w)
        int b = i / HW;
        long long hw = i % HW;
        int h = hw / W;
        int w = hw % W;
        // input base for this b and hw
        const float* in_base = input + (long long)b * C * HW + hw;
        float sum = 0.0f;
        #pragma unroll 8
        for (int c = 0; c < C; ++c) {
            sum += in_base[(long long)c * HW];
        }
        output[i] = sum;
    }
}

// Host launcher for softplus (no PyTorch dependencies, accepts raw pointers and stream)
extern "C" void custom_softplus_launcher(const float* input,
                                         float* output,
                                         long long total_elements,
                                         cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid cap, but kernel uses grid-stride loop
    softplus_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

// Host launcher for sum over dim1
extern "C" void custom_sum_dim1_launcher(const float* input,
                                         float* output,
                                         int N, int C, int H, int W,
                                         cudaStream_t stream) {
    long long HW = (long long)H * W;
    long long total_output = (long long)N * HW;
    int threads = 256;
    int blocks = (int)((total_output + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    sum_dim1_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W);
}
