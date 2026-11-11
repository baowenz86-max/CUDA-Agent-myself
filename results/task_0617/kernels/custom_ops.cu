#include <cuda_runtime.h>

// GLU forward kernel: output = a * sigmoid(b)
// input shape: [B, 2*N, F], output shape: [B, N, F]
// a = input[:, :N, :], b = input[:, N:, :]
__global__ void glu_forward_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int B, int N, int F) {
    int total = B * N * F;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // decompose i into (b, n, f) where n in [0, N), f in [0, F)
        int f = i % F;
        int tmp = i / F;
        int n = tmp % N;
        int b = tmp / N;

        // offset in input for a and b
        int a_idx = ((long long)b * (2 * N) + n) * F + f;
        int b_idx = ((long long)b * (2 * N) + (n + N)) * F + f;

        float a = input[a_idx];
        float bb = input[b_idx];
        // fast sigmoid: 1 / (1 + exp(-x))
        float sig = 1.0f / (1.0f + expf(-bb));
        output[i] = a * sig;
    }
}

// C-interface launcher (no PyTorch dependencies)
extern "C" void glu_forward_launcher(const float* input,
                                     float* output,
                                     int B, int N, int F,
                                     cudaStream_t stream) {
    int total = B * N * F;
    if (total == 0) return;

    const int block_size = 256;
    // use enough blocks to cover the work, but cap to avoid excessive launch overhead
    int grid_size = (total + block_size - 1) / block_size;
    // limit grid to avoid too many blocks, grid-stride loop will handle rest
    if (grid_size > 65535) grid_size = 65535;
    if (grid_size < 1) grid_size = 1;

    glu_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, B, N, F);
}
