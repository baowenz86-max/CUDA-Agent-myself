#include <cuda_runtime.h>

// CUDA kernel for torch.tril on last two dimensions
// Input is a 4D tensor of shape [N, C, H, W], contiguous float32.
// For each element at (n, c, h, w), output = input if w <= h else 0.
// All other dimensions map to linear index using row-major order.
__global__ void tril_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;

    // Compute 4D indices from linear index (row-major)
    int w = idx % W;
    int h = (idx / W) % H;
    // n and c are not needed for the mask
    // int c = (idx / (W * H)) % C;
    // int n = idx / (W * H * C);

    if (w <= h) {
        output[idx] = input[idx];
    } else {
        output[idx] = 0.0f;
    }
}

// CUDA kernel for F.relu: element-wise max(0, x)
__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float val = input[idx];
    output[idx] = val > 0.0f ? val : 0.0f;
}

// Launcher for tril operation
// Parameters: input/output pointers, shape N,C,H,W, CUDA stream
extern "C" void tril_launcher(const float* input, float* output,
                              int N, int C, int H, int W,
                              cudaStream_t stream) {
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    tril_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W);
}

// Launcher for relu operation
// Parameters: input/output pointers, total elements, CUDA stream
extern "C" void relu_launcher(const float* input, float* output,
                              int total_elements,
                              cudaStream_t stream) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}
