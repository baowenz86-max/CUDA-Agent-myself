#include <cuda_runtime.h>

// Per-channel PReLU forward for NCHW float32 tensors.
// weight has length C; output = x if x > 0 else weight[c] * x.
__global__ void prelu_kernel(const float* __restrict__ x,
                             const float* __restrict__ weight,
                             float* __restrict__ out,
                             int N, int C, int H, int W) {
    long long total = static_cast<long long>(N) * C * H * W;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

    // grid-stride loop to cover all elements
    for (; idx < total; idx += stride) {
        // Determine channel index: layout NCHW => index = ((n*C + c)*H + h)*W + w
        long long tmp = idx / (H * W);
        int c = static_cast<int>(tmp % C);
        float val = x[idx];
        float w = weight[c];
        out[idx] = (val > 0.0f) ? val : (w * val);
    }
}

// Launcher function, no PyTorch dependencies.
extern "C" void prelu_forward_launcher(const float* x,
                                      const float* weight,
                                      float* out,
                                      int N, int C, int H, int W,
                                      cudaStream_t stream) {
    long long total = static_cast<long long>(N) * C * H * W;
    if (total <= 0) return;

    const int block_size = 256;
    // Compute grid size with cap, but kernel uses grid-stride loop to handle remainder.
    long long blocks_ll = (total + block_size - 1) / block_size;
    int blocks = static_cast<int>(blocks_ll > 65535 ? 65535 : blocks_ll);

    prelu_kernel<<<blocks, block_size, 0, stream>>>(x, weight, out, N, C, H, W);
}
