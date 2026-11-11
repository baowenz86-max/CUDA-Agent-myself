// custom_ops.cu
// CUDA kernels and launchers for custom operations used in ModelNew.
// No torch/ATen/pybind dependencies; only raw pointers, scalars, and streams.
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Elementwise batchnorm evaluation kernel (eval mode).
// y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// x is assumed to be contiguous in the sense that all N,H,W dimensions are
// contiguous, but strides are provided to support arbitrary memory layout.
// We use a flat index over all elements. For each element, compute channel c
// from the linear index and the strides. This works for any non-overlapping
// layout as long as the channel dimension is contiguous. However, here we
// assume the general case and use strides to compute c for each element.
// To avoid expensive division per element, we can precompute per-channel
// scale and shift on the device? But for simplicity and correctness we
// compute them per block and store in shared memory.
// -----------------------------------------------------------------------------

__global__ void batchnorm_eval_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long y_stride_n, long long y_stride_c, long long y_stride_h, long long y_stride_w,
    long long total_elements)
{
    // Shared memory cache for per-channel scale and shift
    extern __shared__ float smem[];
    float* s_scale = smem;       // size C
    float* s_shift = smem + C;   // size C

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Load per-channel scale and shift into shared memory
    for (int c = tid; c < C; c += block_size) {
        // Use 1/sqrt(var+eps) * weight as scale
        s_scale[c] = weight[c] / sqrtf(running_var[c] + eps);
        s_shift[c] = bias[c] - running_mean[c] * s_scale[c];
    }
    __syncthreads();

    // Compute linear index range for this block
    long long start = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (long long idx = start; idx < total_elements; idx += stride) {
        // Decompose linear index into n,c,h,w using strides and dimensions.
        // This is a general method: iterate over dimensions from outermost to innermost.
        // However, doing division/modulo per element is expensive. Since the input
        // is usually contiguous (N, C, H, W), we can specialize for contiguous case
        // to improve performance. But for correctness with arbitrary strides,
        // we need to handle non-contiguous. We'll implement a faster path for
        // contiguous case (where strides are default) and a general fallback.
        // For this task, input may be non-contiguous (e.g., transposed). We'll implement
        // the general indexing but optimize common contiguous case using inline
        // constants? Actually we can pass a flag or just use the same code with
        // runtime strides; division/modulo cost may be acceptable for 134M elements?
        // The batchnorm loop will be memory-bound, but we can reduce overhead by
        // precomputing the channel index using a different approach: since C is small
        // and H,W are inner dimensions, we can process per (n,c) block of H*W elements
        // and avoid division per element. But with arbitrary strides, we may need
        // to access non-contiguous per element. We'll implement a more efficient
        // kernel that assumes the innermost dimension W is contiguous for both x and y,
        // and H may have a stride >= W. Then we can iterate over n,c,h and for each
        // thread process contiguous W elements? But thread mapping matters.
        // Given complexity, we opt for a simpler yet reasonably efficient approach:
        // grid-stride loop with per-element index calculation using divisions.
        // This is correct and good enough for a memory-bound operation.

        long long rem = idx;
        // Compute n, c, h, w using strides? Actually we need to compute the actual
        // pointer offset based on the strides, not just the linear index in C-order.
        // To compute offset from linear index in the logical tensor, we need to assume
        // the logical layout is C-contiguous (as torch tensors usually have strides
        // that define a mapping from logical multi-index to memory offset). The linear
        // index idx can be decomposed using the strides? The correct way:
        // For a tensor with shape [N,C,H,W] and strides [sn,sc,sh,sw], the memory
        // offset for logical index (n,c,h,w) = n*sn + c*sc + h*sh + w*sw.
        // Given a flat linear index 0 <= idx < N*C*H*W, the logical indices are:
        // n = idx / (C*H*W)
        // c = (idx / (H*W)) % C
        // h = (idx / W) % H
        // w = idx % W
        // This is independent of strides because it's the logical C-contiguous order.
        // Then we compute offset = n*sn + c*sc + h*sh + w*sw.
        // This is correct for any strides as long as the logical dimensions are
        // ordered N,C,H,W.

        long long n = idx / ((long long)C * H * W);
        long long c = (idx / ((long long)H * W)) % C;
        long long h = (idx / W) % H;
        long long w = idx % W;

        long long x_offset = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w;
        long long y_offset = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w;

        float val = x[x_offset];
        float scale = s_scale[c];
        float shift = s_shift[c];
        y[y_offset] = val * scale + shift;
    }
}

// -----------------------------------------------------------------------------
// Elementwise absolute value kernel
// y = |x|, supports strided access
// -----------------------------------------------------------------------------
__global__ void abs_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long total_elements,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long y_stride_n, long long y_stride_c, long long y_stride_h, long long y_stride_w,
    int N, int C, int H, int W)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        // Same decomposition as batchnorm
        long long n = idx / ((long long)C * H * W);
        long long c = (idx / ((long long)H * W)) % C;
        long long h = (idx / W) % H;
        long long w = idx % W;

        long long x_offset = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w;
        long long y_offset = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w;

        y[y_offset] = fabsf(x[x_offset]);
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C")
// -----------------------------------------------------------------------------

// Launch batchnorm_eval_forward.
// Parameters:
//   x: input tensor data pointer (float, on device)
//   weight, bias, running_mean, running_var: float pointers on device, length C
//   eps: epsilon scalar
//   N, C, H, W: dimensions
//   x_strides[4], y_strides[4]: strides for input and output (output has same logical shape as input)
//   stream: CUDA stream
// Returns nothing; output is written to y pointer (must be allocated by caller).
// The binding layer is responsible for allocating y and passing correct pointers.
extern "C" void batchnorm_eval_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long y_stride_n, long long y_stride_c, long long y_stride_h, long long y_stride_w,
    cudaStream_t stream)
{
    long long total_elements = (long long)N * C * H * W;
    if (total_elements == 0) return;

    // Choose block size and grid size. We want enough blocks to fill the GPU.
    int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    // Cap grid size to avoid exceeding max grid dimension (2^31-1).
    if (grid_size > 65535) grid_size = 65535;

    // Shared memory size: 2 * C floats (scale and shift)
    size_t shared_mem = 2 * C * sizeof(float);

    batchnorm_eval_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        x, y, weight, bias, running_mean, running_var, eps,
        N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        y_stride_n, y_stride_c, y_stride_h, y_stride_w,
        total_elements);
}

// Launch abs_forward.
// Parameters:
//   x: input tensor data pointer
//   y: output tensor data pointer (allocated by caller)
//   N, C, H, W: dimensions
//   x_strides[4], y_strides[4]
//   stream: CUDA stream
extern "C" void abs_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long y_stride_n, long long y_stride_c, long long y_stride_h, long long y_stride_w,
    cudaStream_t stream)
{
    long long total_elements = (long long)N * C * H * W;
    if (total_elements == 0) return;

    int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;

    abs_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y, total_elements,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        y_stride_n, y_stride_c, y_stride_h, y_stride_w,
        N, C, H, W);
}
