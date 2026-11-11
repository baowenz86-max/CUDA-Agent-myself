#include <cuda_runtime.h>

// Equivalent to torch.clone(x) followed by torch.rot90(x_clone, k=1, dims=(1,2))
// Input x: float32 pointer, shape [N, C, H, W] where C=3, H=320, W=320
// Output: float32 pointer, contiguous, shape [N, H, C, W]
// The rotation swaps dimensions 1 and 2, and reverses the new dimension 1 (original dim 2).
// Input may be non-contiguous; use strides for safe indexing.
// We assume output is contiguous, allocated by the binding code.
__global__ void rot90_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int64_t total_elements,
                             int64_t N, int64_t C, int64_t H, int64_t W,
                             int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = gridDim.x * blockDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_elements; linear_idx += grid_stride) {
        // Decompose linear index in output layout [N, H, C, W]
        int64_t w = linear_idx % W;
        int64_t tmp = linear_idx / W;
        int64_t c = tmp % C;
        tmp /= C;
        int64_t h = tmp % H;
        int64_t n = tmp / H;

        // Source coordinates: input[n, c, H-1-h, w]
        int64_t src_h = H - 1 - h;
        int64_t src_index = n * in_stride_n + c * in_stride_c + src_h * in_stride_h + w * in_stride_w;

        output[linear_idx] = input[src_index];
    }
}

// Launcher for custom_rot90. The binding code will provide a contiguous output tensor.
// input: float pointer to input data (may be non-contiguous, strides supplied)
// output: float pointer to contiguous output data (pre-allocated)
// N, C, H, W: dimensions of input
// in_stride_n, in_stride_c, in_stride_h, in_stride_w: strides of input in elements
// stream: CUDA stream to launch on
extern "C" void custom_rot90_launcher(const float* input,
                                       float* output,
                                       int64_t N, int64_t C, int64_t H, int64_t W,
                                       int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
                                       cudaStream_t stream) {
    int64_t total_elements = N * H * C * W;  // output elements count

    // Choose block size and grid size
    const int threads = 256;
    int64_t blocks = (total_elements + threads - 1) / threads;
    // Cap grid to avoid launch failure, but no need to cap for this size (blocks~614400 < 2^31-1)
    // Use grid-stride loop inside kernel anyway for flexibility.
    if (blocks > 2147483647) blocks = 2147483647; // safety cap (not reached)

    rot90_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        input, output, total_elements, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, would throw; here just ignore because launcher has no exception mechanism.
        // Binding code should check after calling launcher.
    }
}
