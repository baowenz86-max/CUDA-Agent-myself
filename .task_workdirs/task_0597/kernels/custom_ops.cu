#include <cuda_runtime.h>

// Channel shuffle kernel: output[n,c,h,w] = input[n, perm(c), h, w]
// where perm(c) = (c % groups) * (C / groups) + (c / groups)
// Input and output are contiguous NCHW tensors.
__global__ void channel_shuffle_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int total_elements,
    int C,
    int HW,
    int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int channels_per_group = C / groups;

    for (int i = idx; i < total_elements; i += stride) {
        // Decompose flattened index into n, c, hw
        int hw = i % HW;
        int c = (i / HW) % C;
        int n = i / (HW * C);

        // Source channel index
        int src_c = (c % groups) * channels_per_group + (c / groups);

        // Source flattened index
        int src_idx = ((n * C + src_c) * HW) + hw;

        output[i] = input[src_idx];
    }
}

extern "C" void channel_shuffle_launcher(
    float* output,
    const float* input,
    int N,
    int C,
    int H,
    int W,
    int groups,
    cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || groups <= 0) {
        return;
    }

    int HW = H * W;
    long long total_elements_ll = 1LL * N * C * HW;
    if (total_elements_ll <= 0) {
        return;
    }

    // Use int for indexing; total elements for contract is 268435456 < INT_MAX, but we guard generally.
    // If total elements may exceed INT_MAX, we would need 64-bit indexing; but for this task it's within int.
    int total_elements = static_cast<int>(total_elements_ll);

    int block_size = 256;
    // Calculate grid size capped at 65535 (max grid x dimension for simplicity); use grid-stride loop.
    int grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    channel_shuffle_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, total_elements, C, HW, groups);
}
