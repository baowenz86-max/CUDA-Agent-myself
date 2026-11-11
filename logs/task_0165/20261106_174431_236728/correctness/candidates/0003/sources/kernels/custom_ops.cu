#include <cuda_runtime.h>

// Numerically stable log_softmax along a specified dimension.
// Template instantiated for float only (as used by the model).
// The input is assumed contiguous (as guaranteed by the binding layer).
// dim is the dimension along which to compute log_softmax, 0 <= dim < ndims.
// Output has the same shape and strides as input.
template <typename scalar_t>
__global__ void log_softmax_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int outer_size,
    int reduction_size,
    int inner_size,
    long total_positions)
{
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;

    for (; idx < total_positions; idx += stride) {
        int inner = idx % inner_size;
        long outer = idx / inner_size;

        const scalar_t* in_base = input + outer * (long)reduction_size * inner_size + inner;
        scalar_t* out_base = output + outer * (long)reduction_size * inner_size + inner;

        scalar_t max_val = in_base[0];
        for (int r = 1; r < reduction_size; ++r) {
            scalar_t v = in_base[(long)r * inner_size];
            if (v > max_val) max_val = v;
        }

        scalar_t sum_exp = 0;
        for (int r = 0; r < reduction_size; ++r) {
            scalar_t v = in_base[(long)r * inner_size] - max_val;
            sum_exp += expf(v);
        }
        scalar_t log_sum_exp = logf(sum_exp);

        for (int r = 0; r < reduction_size; ++r) {
            scalar_t v = in_base[(long)r * inner_size] - max_val - log_sum_exp;
            out_base[(long)r * inner_size] = v;
        }
    }
}

// Launcher for log_softmax. dims is an array of size ndims with the tensor shape.
// All dimensions are positive and fit in int; total elements can be larger.
// `dim` is the reduction dimension (same as Python dim).
extern "C" void log_softmax_forward_launcher(
    float* output,
    const float* input,
    const int64_t* dims,
    int ndims,
    int dim,
    cudaStream_t stream)
{
    if (dim < 0) dim += ndims;

    long outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= dims[i];
    long reduction_size = dims[dim];
    long inner_size = 1;
    for (int i = dim + 1; i < ndims; ++i) inner_size *= dims[i];

    long total_positions = outer_size * inner_size;
    if (total_positions == 0) return;

    const int block = 256;
    long blocks_needed = (total_positions + block - 1) / block;
    int grid = (int)min((long)65535, blocks_needed);

    log_softmax_kernel<float><<<grid, block, 0, stream>>>(
        output, input,
        (int)outer_size,
        (int)reduction_size,
        (int)inner_size,
        total_positions);
}

// ======================= ConvTranspose1d forward =======================
// Direct computation of transpose convolution without cuDNN.
// Assumes input is contiguous, float32, shape [N, C_in, L_in]
// weight shape [C_in, C_out, K]
// bias optional, shape [C_out]
// stride, padding: ints
// output shape [N, C_out, L_out]
__global__ void conv_transpose1d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, // may be null
    int N, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding,
    int L_out,
    bool has_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int l = idx % L_out;
    int tmp = idx / L_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k = 0; k < K; ++k) {
            int l_in_pos = l + padding - k;
            if (l_in_pos >= 0 && l_in_pos % stride == 0) {
                int l_in = l_in_pos / stride;
                if (l_in < L_in) {
                    acc += input[((n * C_in + c_in) * L_in) + l_in] *
                           weight[((c_in * C_out + c_out) * K) + k];
                }
            }
        }
    }
    if (has_bias) acc += bias[c_out];
    output[idx] = acc;
}

extern "C" void conv_transpose1d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding,
    int L_out,
    bool has_bias,
    cudaStream_t stream)
{
    int total = N * C_out * L_out;
    const int block = 256;
    int grid = (total + block - 1) / block;
    conv_transpose1d_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, bias,
        N, C_in, L_in, C_out, K,
        stride, padding, L_out, has_bias);
}

// ======================= ConvTranspose2d forward =======================
// Direct computation, assumes contiguous inputs.
// input shape [N, C_in, H_in, W_in]
// weight shape [C_in, C_out, K_h, K_w]
// bias optional [C_out]
// stride, padding, output_padding
// output shape [N, C_out, H_out, W_out]
__global__ void conv_transpose2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int padding, int output_padding,
    int H_out, int W_out,
    bool has_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int tmp = idx / W_out;
    int h = tmp % H_out;
    tmp /= H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in_pos = h + padding - kh;
                int w_in_pos = w + padding - kw;
                if (h_in_pos >= 0 && h_in_pos % stride == 0 &&
                    w_in_pos >= 0 && w_in_pos % stride == 0) {
                    int h_in = h_in_pos / stride;
                    int w_in = w_in_pos / stride;
                    if (h_in < H_in && w_in < W_in) {
                        acc += input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in] *
                               weight[((c_in * C_out + c_out) * K_h + kh) * K_w + kw];
                    }
                }
            }
        }
    }
    if (has_bias) acc += bias[c_out];
    output[idx] = acc;
}

extern "C" void conv_transpose2d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int padding, int output_padding,
    int H_out, int W_out,
    bool has_bias,
    cudaStream_t stream)
{
    int total = N * C_out * H_out * W_out;
    const int block = 256;
    int grid = (total + block - 1) / block;
    conv_transpose2d_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, bias,
        N, C_in, H_in, W_in, C_out, K_h, K_w,
        stride, padding, output_padding, H_out, W_out, has_bias);
}
