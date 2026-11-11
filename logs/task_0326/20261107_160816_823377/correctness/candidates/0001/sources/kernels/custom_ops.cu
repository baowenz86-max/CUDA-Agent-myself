#include <cuda_runtime.h>

// CUDA kernel for hardsigmoid: clamp(x + 3, 0, 6) / 6
__global__ void hardsigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float x = input[idx];
        float y = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
        output[idx] = y;
    }
}

// CUDA kernel for mean over last two dimensions.
// Input shape: [N, C, H, W], output shape: [N, C].
// stride_n, stride_c, stride_h, stride_w are in elements.
__global__ void mean_last_2_dims_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W,
    long long stride_n,
    long long stride_c,
    long long stride_h,
    long long stride_w)
{
    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.x;
    if (batch_idx >= N || channel_idx >= C) return;

    long long base = (long long)batch_idx * stride_n + (long long)channel_idx * stride_c;
    long long total = (long long)H * W;
    double sum = 0.0;

    // Loop over all elements in the (batch, channel) slice
    // Use thread coarsening: each thread handles some rows.
    // For simplicity, launch one block with 256 threads per (b,c) and use grid-stride loop within the slice.
    // Alternatively, launch enough threads and use atomicAdd, but we want deterministic result.
    // Here we use a single thread per output element to compute sum sequentially? Not efficient.
    // Instead, use block reduce within one block per (b,c).
    // Launch config: blockDim.x=256, gridDim.x=C, gridDim.y=N.
    // Each block reduces a (H*W) slice using shared memory.
    extern __shared__ float shared_sum[];
    int tid = threadIdx.x;
    float local_sum = 0.0f;

    // Grid-stride loop over the flattened slice elements
    for (long long i = tid; i < total; i += blockDim.x) {
        // Convert linear i to (h,w) or just directly use linear offset assuming contiguous?
        // Must handle arbitrary strides: need h = i / W, w = i % W
        // This could be expensive; but we only have fixed W=64, so maybe use division.
        // Better: loop over h and w explicitly for simplicity, but less coalesced.
        // Since input may not be contiguous, performance may suffer, but correctness first.
        // For coalescing, it would be better to make input contiguous before mean.
        // However the launcher could handle non-contiguous by making a contiguous copy? But we cannot allocate here.
        // We'll assume that after hardsigmoid_forward, the output tensor is contiguous (the binding will allocate contiguous output).
        // Actually hardsigmoid_forward must preserve shape but output may be non-contiguous if input is non-contiguous? But binding likely creates empty_like which is contiguous? Not guaranteed.
        // But for the mean kernel, we can read input with stride by computing h and w from linear index:
        long long h = i / W;
        long long w = i - h * W;
        long long offset = base + h * stride_h + w * stride_w;
        local_sum += input[offset];
    }

    // Block reduction
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[(long long)batch_idx * C + channel_idx] = (float)(shared_sum[0] / (double)total);
    }
}

// Launcher for hardsigmoid_forward: assumes input is contiguous (since no stride info needed for elementwise)
// But contract says input may be non-contiguous. We can handle by treating input as contiguous pointer only if binding passes
// a contiguous view? The launcher here gets only raw pointers and total elements; if non-contiguous, this kernel would be wrong.
// However, the binding can enforce contiguity before calling launcher, or we can handle stride in kernel.
// To strictly follow contract "input may be non-contiguous", we should handle strides in kernel.
// But elementwise operation only depends on value at each logical index, not on layout; if we pass the logical total and raw pointer,
// it works only contiguous. To support non-contiguous, we need strides. Let's modify launcher to receive strides.
// Since the binding will be generated later, we can define a launcher with stride parameters and document them.
// For simplicity and to align with likely binding, we could accept a bool is_contiguous and if false maybe use a different kernel.
// But we cannot allocate. Better to implement stride-aware elementwise kernel.
// We'll define hardsigmoid_forward_launcher with shape and strides:
// hardsigmoid_forward_launcher(input_ptr, output_ptr, N, C, H, W, stride_n, stride_c, stride_h, stride_w, stream)
// Then kernel uses linear index and convert to n,c,h,w via division/modulo, then compute offset with strides.
// This may be slower than contiguous version, but correct.
// Given the input likely contiguous (from randn), we can also add a fast path if strides match contiguous pattern.
// But launcher doesn't know if it's contiguous; it can check stride values.
// We'll implement two kernels: one contiguous, one strided, choose at runtime inside launcher.

// Strided elementwise kernel
__global__ void hardsigmoid_strided_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W,
    long long stride_n,
    long long stride_c,
    long long stride_h,
    long long stride_w)
{
    long long total = (long long)N * C * H * W;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;
    for (long long i = idx; i < total; i += stride) {
        // Convert i to coordinates
        long long w = i % W;
        long long tmp = i / W;
        long long h = tmp % H;
        tmp = tmp / H;
        long long c = tmp % C;
        long long n = tmp / C;

        long long src_offset = n * stride_n + c * stride_c + h * stride_h + w * stride_w;
        float x = input[src_offset];
        float y = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
        output[src_offset] = y; // output same layout as input (we assume binding allocates same strides)
    }
}

// Contiguous elementwise kernel (fast path)
__global__ void hardsigmoid_contiguous_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long total)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;
    for (long long i = idx; i < total; i += stride) {
        float x = input[i];
        float y = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
        output[i] = y;
    }
}

// Extern C launchers
extern "C" void hardsigmoid_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    long long stride_n,
    long long stride_c,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    if (total == 0) return;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid limit for x dimension

    // Check if strides are contiguous: stride_w=1, stride_h=W, stride_c=H*W, stride_n=C*H*W
    bool contiguous = (stride_w == 1 && stride_h == W && stride_c == (long long)H * W && stride_n == (long long)C * H * W);
    if (contiguous) {
        hardsigmoid_contiguous_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
    } else {
        hardsigmoid_strided_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W, stride_n, stride_c, stride_h, stride_w);
    }
}

extern "C" void mean_last_2_dims_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    long long stride_n,
    long long stride_c,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream)
{
    if (N == 0 || C == 0) return;

    dim3 grid(C, N);
    int blockDim = 256;
    int sharedMemSize = blockDim * sizeof(float);

    mean_last_2_dims_kernel<<<grid, blockDim, sharedMemSize, stream>>>(
        input, output, N, C, H, W, stride_n, stride_c, stride_h, stride_w);
}
