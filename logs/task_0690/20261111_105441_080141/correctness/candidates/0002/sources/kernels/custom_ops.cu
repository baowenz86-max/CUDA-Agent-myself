#include <cuda_runtime.h>
#include <cmath>

// Kernel for concatenating two tensors along channel dimension (dim=1)
// Inputs: contiguous float32 tensors of shape [N, C, H, W]
// Output: contiguous float32 tensor of shape [N, 2*C, H, W]
// Mapping: output[n, c_out, h, w] = (c_out < C) ? input1[n, c_out, h, w] : input2[n, c_out-C, h, w]
__global__ void cat_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * 2 * C * H * W;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        int w = i % W;
        int tmp = i / W;
        int h = tmp % H;
        tmp = tmp / H;
        int c_out = tmp % (2 * C);
        int n = tmp / (2 * C);

        if (c_out < C) {
            int in_idx = ((n * C + c_out) * H + h) * W + w;
            output[i] = input1[in_idx];
        } else {
            int c = c_out - C;
            int in_idx = ((n * C + c) * H + h) * W + w;
            output[i] = input2[in_idx];
        }
    }
}

// Kernel for elementwise cosine
__global__ void cos_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        output[i] = cosf(input[i]);
    }
}

// Host launcher for cat_forward
extern "C" void cat_forward_launcher(
    float* output,
    const float* input1,
    const float* input2,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total_elements = N * 2 * C * H * W;
    if (total_elements <= 0) return;

    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cat_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, input1, input2, N, C, H, W);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore; binding layer may check synchronization.
    }
}

// Host launcher for cos_forward
extern "C" void cos_forward_launcher(
    float* output,
    const float* input,
    int total_elements,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;

    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cos_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore for now; binding can check.
    }
}
