#include <cuda_runtime.h>
#include <cstdint>

// Softmax along dim=1 for 4D float32 tensor with arbitrary strides.
// Input: x pointer, output pointer, dims[4], strides[4], total elements, stream.
// We assume dims[1] is the class dimension, and we launch one thread per (n, h, w) position.
__global__ void softmax_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    // Grid-stride over planes (n, h, w)
    long long total_planes = (long long)N * H * W;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_planes; idx += (long long)gridDim.x * blockDim.x) {
        int n = idx / ((long long)H * W);
        int rem = idx % ((long long)H * W);
        int h = rem / W;
        int w = rem % W;

        // Base pointers for this plane
        const float* base_x = x + n * stride_n + h * stride_h + w * stride_w;
        float* base_out = out + n * out_stride_n + h * out_stride_h + w * out_stride_w;

        // Find max along C
        float max_val = base_x[0 * stride_c];
        for (int c = 1; c < C; ++c) {
            float val = base_x[c * stride_c];
            if (val > max_val) max_val = val;
        }

        // Compute sum of exp(x - max)
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += __expf(base_x[c * stride_c] - max_val);
        }

        // Write softmax values
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < C; ++c) {
            base_out[c * out_stride_c] = __expf(base_x[c * stride_c] - max_val) * inv_sum;
        }
    }
}

// Hardtanh element-wise with arbitrary ndim and strides, clamp to [-1,1] matching F.hardtanh defaults.
__global__ void hardtanh_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    long long total_elems,
    int ndim,
    const int64_t* __restrict__ dims,
    const int64_t* __restrict__ istride, // input strides in elements
    const int64_t* __restrict__ ostride  // output strides in elements
)
{
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elems; idx += (long long)gridDim.x * blockDim.x) {
        // Recover multidimensional index from linear idx
        long long rem = idx;
        long long in_off = 0;
        long long out_off = 0;
        // We rely on ndim being small, e.g., <= 8
        for (int d = 0; d < ndim; ++d) {
            int coord = rem % dims[d];
            rem /= dims[d];
            in_off += coord * istride[d];
            out_off += coord * ostride[d];
        }
        float val = x[in_off];
        out[out_off] = fminf(fmaxf(val, -1.0f), 1.0f);
    }
}

extern "C" void softmax_dim1_forward_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total_planes = (long long)N * H * W;
    if (total_planes == 0) return;

    int threads = 256;
    int blocks = (int)((total_planes + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid-stride loop will handle the rest
    softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(
        x, out, N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w
    );
}

extern "C" void hardtanh_forward_launcher(
    const float* x,
    float* out,
    long long total_elems,
    int ndim,
    const int64_t* dims,
    const int64_t* istride,
    const int64_t* ostride,
    cudaStream_t stream)
{
    if (total_elems == 0) return;
    int threads = 256;
    int blocks = (int)((total_elems + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    hardtanh_kernel<<<blocks, threads, 0, stream>>>(
        x, out, total_elems, ndim, dims, istride, ostride
    );
}
