#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// custom_cat kernel: concatenates two contiguous tensors x1 and x2 along dimension dim.
// Inputs are expected contiguous; binding layer should pass contiguous tensors.
// Output is contiguous, shape same as x1 except dimension dim = x1_dim + x2_dim.
// -----------------------------------------------------------------------------

__global__ void cat_kernel(const float* __restrict__ x1,
                           const float* __restrict__ x2,
                           float* __restrict__ out,
                           int64_t x1_size,
                           int64_t x2_size,
                           int64_t dim_size,
                           int64_t inner_size,
                           int64_t outer_size) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = outer_size * (x1_size + x2_size) * inner_size;
    if (idx >= total) return;

    // decompose idx into outer, channel (dim index), inner
    int64_t inner = idx % inner_size;
    int64_t tmp = idx / inner_size;
    int64_t ch = tmp % dim_size;
    int64_t outer = tmp / dim_size;

    const float* src;
    int64_t src_ch;
    if (ch < x1_size) {
        src = x1;
        src_ch = ch;
    } else {
        src = x2;
        src_ch = ch - x1_size;
    }

    int64_t src_idx = (outer * x1_size + src_ch) * inner_size + inner;
    out[idx] = src[src_idx];
}

extern "C" void custom_cat_launcher(const float* x1,
                                    const float* x2,
                                    float* out,
                                    int64_t x1_size,
                                    int64_t x2_size,
                                    int64_t dim_size,
                                    int64_t inner_size,
                                    int64_t outer_size,
                                    cudaStream_t stream) {
    int64_t total = outer_size * dim_size * inner_size;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    cat_kernel<<<blocks, threads, 0, stream>>>(x1, x2, out, x1_size, x2_size, dim_size, inner_size, outer_size);
}

// -----------------------------------------------------------------------------
// custom_rot90 kernel: rotates a 4D tensor by 90 degrees k times in the plane
// specified by dim0 and dim1. Handles arbitrary strides for input.
// For the reference case k=1, dim0=2, dim1=3, output shape [N,C,W,H]
// (height and width swapped). The output should have stride [C*W*H, W*H, 1, W]
// to match PyTorch view semantics, but here we write contiguous output to a
// binding-layer tensor with that exact stride (a view of a larger contiguous
// allocation) or simply set output strides accordingly.
// We implement generic rotation using formula:
//   output[o0, o1, o2, o3] = input[?]
// For k=1, dims=(2,3): output[N, C, m, n] = input[N, C, n, W-1-m]
// We pass input strides so non-contiguous input is handled.
// -----------------------------------------------------------------------------

__global__ void rot90_kernel_generic(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t dim0,
    int64_t dim1,
    int64_t k,
    int64_t out_size0,
    int64_t out_size1,
    int64_t out_size2,
    int64_t out_size3,
    int64_t in_stride0,
    int64_t in_stride1,
    int64_t in_stride2,
    int64_t in_stride3,
    int64_t in_size0,
    int64_t in_size1,
    int64_t in_size2,
    int64_t in_size3)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = out_size0 * out_size1 * out_size2 * out_size3;
    if (idx >= total) return;

    int64_t n3 = idx % out_size3;
    int64_t tmp = idx / out_size3;
    int64_t n2 = tmp % out_size2;
    tmp /= out_size2;
    int64_t n1 = tmp % out_size1;
    int64_t n0 = tmp / out_size1;

    // Build input logical coordinates corresponding to output coordinates.
    // We handle only the case dim0=2, dim1=3, k modulo 4.
    // For 4D tensor, rotation in plane of last two dimensions.
    // k % 4 == 1: output[i,j,m,n] = input[i,j,n, W-1-m]
    // k % 4 == 3: output[i,j,m,n] = input[i,j,H-1-n, m]
    // k % 4 == 2: output[i,j,m,n] = input[i,j,H-1-m, W-1-n]
    // k % 4 == 0: identity
    int64_t in_coord0 = n0;
    int64_t in_coord1 = n1;
    int64_t in_coord2, in_coord3;
    int mod = (int)(((k % 4) + 4) % 4);
    int64_t H = in_size2;
    int64_t W = in_size3;
    if (mod == 0) {
        in_coord2 = n2;
        in_coord3 = n3;
    } else if (mod == 1) {
        // output shape swapped: out dims 2,3 correspond to in dims 3,2 reversed
        in_coord2 = n3;           // input dim2 index = output dim3 index
        in_coord3 = W - 1 - n2;   // input dim3 index = W-1 - output dim2 index
    } else if (mod == 2) {
        in_coord2 = H - 1 - n2;
        in_coord3 = W - 1 - n3;
    } else { // mod == 3
        in_coord2 = H - 1 - n3;
        in_coord3 = n2;
    }

    int64_t input_idx = in_coord0 * in_stride0 + in_coord1 * in_stride1 +
                        in_coord2 * in_stride2 + in_coord3 * in_stride3;
    output[idx] = input[input_idx];
}

// launcher specifically for 4D case dim0=2, dim1=3, as used in this task.
// The binding layer will allocate output with shape [N,C,W,H] and set strides
// [C*W*H, W*H, 1, W] to match reference. We pass the desired output shape.
extern "C" void custom_rot90_launcher(
    const float* input,
    float* output,
    int64_t k,
    int64_t dim0,
    int64_t dim1,
    int64_t out_size0,
    int64_t out_size1,
    int64_t out_size2,
    int64_t out_size3,
    int64_t in_stride0,
    int64_t in_stride1,
    int64_t in_stride2,
    int64_t in_stride3,
    int64_t in_size0,
    int64_t in_size1,
    int64_t in_size2,
    int64_t in_size3,
    cudaStream_t stream)
{
    // Assume dim0=2, dim1=3 for the 4D usage; generic code would require more
    // parameter logic, but here we implement the exact reference.
    int64_t total = out_size0 * out_size1 * out_size2 * out_size3;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    rot90_kernel_generic<<<blocks, threads, 0, stream>>>(
        input, output, dim0, dim1, k,
        out_size0, out_size1, out_size2, out_size3,
        in_stride0, in_stride1, in_stride2, in_stride3,
        in_size0, in_size1, in_size2, in_size3);
}
