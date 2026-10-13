#include <cuda_runtime.h>

// ELU activation: f(x) = x if x > 0 else alpha * (exp(x) - 1), alpha=1.0
__global__ void elu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : (expf(x) - 1.0f);
    }
}

// Mean over channel dimension (dim=1) with keepdim=True.
// Input shape: [N, C, H, W] (C=3), output shape: [N, 1, H, W].
// This kernel assumes input is contiguous, but can handle non-contiguous by using strides if needed.
// For simplicity and performance, we use the contiguous case: input is contiguous as per contract.
__global__ void mean_dim1_keepdim_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * H * W;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            // input is contiguous with shape [N, C, H, W]
            long long input_idx = ((long long)n * C + c) * (H * W) + hw;
            sum += input[input_idx];
        }
        output[i] = sum / C;
    }
}

// Add scalar (broadcast) to every element of input tensor.
// scalar is a single float value at scalar_ptr[0].
__global__ void add_scalar_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ scalar,
    long long numel)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    float s = scalar[0];
    for (long long i = idx; i < numel; i += stride) {
        output[i] = input[i] + s;
    }
}

// Launcher for elementwise ELU.
// output and input are raw pointers to float32 CUDA memory.
extern "C" void elu_forward_launcher(
    float* output,
    const float* input,
    long long numel,
    cudaStream_t stream)
{
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    if (blocks_ll > 65535) blocks_ll = 65535; // grid cap, grid-stride loop handles rest
    int blocks = (int)blocks_ll;
    elu_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}

// Launcher for mean over dim=1 with keepdim.
// Input must be contiguous with shape [N, C, H, W].
extern "C" void mean_dim1_keepdim_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int threads = 256;
    long long total = (long long)N * H * W;
    long long blocks_ll = (total + threads - 1) / threads;
    if (blocks_ll > 65535) blocks_ll = 65535;
    int blocks = (int)blocks_ll;
    mean_dim1_keepdim_kernel<<<blocks, threads, 0, stream>>>(output, input, N, C, H, W);
}

// Launcher for add scalar.
extern "C" void add_scalar_launcher(
    float* output,
    const float* input,
    const float* scalar,
    long long numel,
    cudaStream_t stream)
{
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    if (blocks_ll > 65535) blocks_ll = 65535;
    int blocks = (int)blocks_ll;
    add_scalar_kernel<<<blocks, threads, 0, stream>>>(output, input, scalar, numel);
}
