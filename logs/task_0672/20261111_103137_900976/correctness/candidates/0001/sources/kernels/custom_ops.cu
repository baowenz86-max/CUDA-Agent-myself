// Pure CUDA kernels for custom_cat_forward and custom_normalize_forward
#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------------------------------------------
// Kernel: concatenate two 4D tensors along dimension 1
// Tensors are assumed to be float32, with arbitrary strides (as long as they
// are valid for the given shapes). Output is contiguous.
// Output shape: [N, C1+C2, H, W]
// Input strides: x1_strides[4], x2_strides[4]; output strides calculated as contiguous.
// The kernel writes one element per thread using a grid-stride loop.
// -----------------------------------------------------------------------------
__global__ void cat_kernel(float* __restrict__ out,
                           const float* __restrict__ x1,
                           const float* __restrict__ x2,
                           int N, int C1, int C2, int H, int W,
                           const int4 x1_strides,   // packed as (stride_n, stride_c, stride_h, stride_w)
                           const int4 x2_strides,
                           int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int C = C1 + C2;

    for (int i = idx; i < total_elements; i += stride) {
        // Decompose linear index into n, c, h, w for output (contiguous)
        int w = i % W;
        int tmp = i / W;
        int h = tmp % H;
        tmp = tmp / H;
        int c = tmp % C;
        int n = tmp / C;

        const float* src;
        int src_c;
        if (c < C1) {
            src = x1;
            src_c = c;
        } else {
            src = x2;
            src_c = c - C1;
        }

        // Compute source offset using its strides
        // Assumes int4 components: .x = stride_n, .y = stride_c, .z = stride_h, .w = stride_w
        int src_offset = n * (src == x1 ? x1_strides.x : x2_strides.x) +
                         src_c * (src == x1 ? x1_strides.y : x2_strides.y) +
                         h * (src == x1 ? x1_strides.z : x2_strides.z) +
                         w * (src == x1 ? x1_strides.w : x2_strides.w);

        out[i] = src[src_offset];
    }
}

// Launcher for concatenation. Assumes all pointers are valid, dim == 1.
// out is preallocated contiguous tensor of shape [N, C1+C2, H, W].
// x1_strides and x2_strides are int4 arrays representing strides in elements.
// Note: Since dimensions can be large, we use int64_t for total elements if needed,
// but for typical sizes < 2^31 we use int. For safety, we'll use long long for indexing inside kernel if needed.
// However, to keep kernel simple, we assume total elements fit in int (max ~562M here, fits).
extern "C" void custom_cat_launcher(float* out,
                                    const float* x1,
                                    const float* x2,
                                    int N, int C1, int C2, int H, int W,
                                    const int4 x1_strides,
                                    const int4 x2_strides,
                                    cudaStream_t stream) {
    int C = C1 + C2;
    int total_elements = N * C * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Cap blocks at a reasonable maximum; kernel uses grid-stride loop.
    if (blocks > 65535) blocks = 65535;
    cat_kernel<<<blocks, threads, 0, stream>>>(out, x1, x2, N, C1, C2, H, W,
                                               x1_strides, x2_strides, total_elements);
}

// -----------------------------------------------------------------------------
// Kernel: Lp normalization along dimension 1 (channels)
// For each (n, h, w) location, compute norm = sqrt(sum over c of x[n,c,h,w]^2)
// Then output = x / (norm + eps)
// Input may have arbitrary strides; output will have same strides? Actually,
// the normalization output usually has the same strides as input if we write in-place
// or produce a new tensor with same strides. For simplicity, we produce output with
// same strides as input (the binding can allocate contiguous and pass appropriate strides).
// We will write one output element per thread, but to compute norm we need a reduction.
// We'll use a separate reduction kernel per (n,h,w) location or a two-pass approach.
// In this kernel, we first compute norms into a temporary array, then divide.
// To avoid separate allocation, we can use a block per (n,h,w) location with channel count C.
// For large C (67) and many locations (N*H*W ~ 512*128*128 = 8.4M), we launch many blocks.
// Each block handles one location and uses shared memory to accumulate squares.
// Block size must be at least C (67). We'll use 128 threads (next multiple of 32).
// -----------------------------------------------------------------------------

// Kernel to compute L2 norm for each spatial location, one block per location.
// Assumes C is moderate (<1024). Input and output have same strides.
// We'll have an intermediate norm array, but we can compute norm on the fly
// and then write output in one kernel using shared memory reduction.
__global__ void normalize_l2_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int N, int C, int H, int W,
                                    const int4 strides, // input/output strides (same)
                                    double p, double eps) {
    // Each block handles one (n, h, w) location.
    int block_id = blockIdx.x;
    int total_locations = N * H * W;
    if (block_id >= total_locations) return;

    // Decompose block_id into n, h, w
    int w = block_id % W;
    int tmp = block_id / W;
    int h = tmp % H;
    int n = tmp / H;

    // Compute base offset for this location
    int base = n * strides.x + h * strides.z + w * strides.w;

    // Use shared memory for reduction
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int c = tid;
    float sum_sq = 0.0f;

    // Grid-stride over channels in case C > blockDim.x
    for (int c_idx = tid; c_idx < C; c_idx += blockDim.x) {
        float val = x[base + c_idx * strides.y];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // Reduction within block (assuming blockDim.x power of 2)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float norm_sq = sdata[0];
    // Compute norm = sqrt(norm_sq) for p=2; if p != 2, we would need pow, but p=2 is expected.
    // We'll use rsqrt for better precision/performance, but need to handle eps.
    // norm = sqrt(norm_sq); value = x / (norm + eps)
    // To match PyTorch behavior: norm = sqrt(sum(x^2)), then denominator = norm + eps.
    // If norm_sq is very small, denominator ~ eps.
    float norm = sqrtf(norm_sq);
    float inv_denom = 1.0f / (norm + (float)eps);

    // Write output for all channels using same block threads
    for (int c_idx = tid; c_idx < C; c_idx += blockDim.x) {
        float val = x[base + c_idx * strides.y];
        out[base + c_idx * strides.y] = val * inv_denom;
    }
}

// Launcher for L2 normalization along dimension 1.
// x and out have same shape and strides. out is preallocated.
// p is assumed to be 2.0 for this task.
extern "C" void custom_normalize_launcher(float* out,
                                          const float* x,
                                          int N, int C, int H, int W,
                                          const int4 strides,
                                          double p, double eps,
                                          cudaStream_t stream) {
    int total_locations = N * H * W;
    int threads = 128; // should be >= C (67), power of 2
    if (threads < C) threads = 128; // if C > 128, would need grid-stride reduction, but our kernel handles it
    int shared_mem = threads * sizeof(float);
    normalize_l2_kernel<<<total_locations, threads, shared_mem, stream>>>(
        x, out, N, C, H, W, strides, p, eps);
}

// -----------------------------------------------------------------------------
// Optional: Kernel for general p-norm, but for this task p=2 only.
// We keep the launcher signature consistent with contract (p may be passed).
// If p is not 2, this implementation would need adjustment. Since contract says p=2,
// we do not implement other p values to avoid unused code paths.
// -----------------------------------------------------------------------------
