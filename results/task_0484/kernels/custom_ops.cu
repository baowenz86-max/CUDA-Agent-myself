#include <cuda_runtime.h>
#include <cfloat>

// Forward declarations of kernels
__global__ void mean_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t is0, int64_t is1, int64_t is2, int64_t is3,
    int64_t iss0, int64_t iss1, int64_t iss2, int64_t iss3,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t oss0, int64_t oss1, int64_t oss2, int64_t oss3,
    int red_dim,
    int64_t total_out);

__global__ void softmax_lastdim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    int64_t str0, int64_t str1, int64_t str2, int64_t str3,
    int64_t outer_size,
    int64_t W);

// Launcher for custom_mean_dim
extern "C" void custom_mean_dim_launcher(
    const float* input,
    float* output,
    const int64_t* in_shape,
    const int64_t* in_strides,
    const int64_t* out_shape,
    const int64_t* out_strides,
    int64_t dim,
    int64_t ndim,
    cudaStream_t stream)
{
    int64_t total_out = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        total_out *= out_shape[i];
    }

    int threads = 256;
    int blocks = (int)((total_out + threads - 1) / threads);

    int64_t is[4] = {1,1,1,1}, iss[4] = {0,0,0,0}, os[4] = {1,1,1,1}, oss[4] = {0,0,0,0};
    for (int64_t i=0; i<ndim; ++i) {
        is[i] = in_shape[i];
        iss[i] = in_strides[i];
        os[i] = out_shape[i];
        oss[i] = out_strides[i];
    }
    int red_dim = (int)dim;

    dim3 grid(blocks);
    dim3 block(threads);

    mean_dim_kernel<<<grid, block, 0, stream>>>(
        input, output,
        is[0], is[1], is[2], is[3],
        iss[0], iss[1], iss[2], iss[3],
        os[0], os[1], os[2], os[3],
        oss[0], oss[1], oss[2], oss[3],
        red_dim, total_out);
}

__global__ void mean_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t is0, int64_t is1, int64_t is2, int64_t is3,
    int64_t iss0, int64_t iss1, int64_t iss2, int64_t iss3,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t oss0, int64_t oss1, int64_t oss2, int64_t oss3,
    int red_dim,
    int64_t total_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int64_t is[4] = {is0, is1, is2, is3};
    int64_t iss[4] = {iss0, iss1, iss2, iss3};
    int64_t os[4] = {os0, os1, os2, os3};
    int64_t oss[4] = {oss0, oss1, oss2, oss3};

    int64_t out_coords[4];
    int64_t tmp = idx;
    for (int d = 3; d >= 0; --d) {
        if (oss[d] != 0) {
            out_coords[d] = tmp % os[d];
            tmp /= os[d];
        } else {
            out_coords[d] = 0;
        }
    }

    int64_t in_offset = 0;
    for (int d = 0; d < 4; ++d) {
        if (d == red_dim) {
            in_offset += 0;
        } else {
            in_offset += out_coords[d] * iss[d];
        }
    }

    float sum = 0.0f;
    int64_t red_size = is[red_dim];
    for (int64_t i = 0; i < red_size; ++i) {
        sum += input[in_offset + i * iss[red_dim]];
    }
    float mean = sum / (float)red_size;

    int64_t out_offset = 0;
    for (int d = 0; d < 4; ++d) {
        out_offset += out_coords[d] * oss[d];
    }
    output[out_offset] = mean;
}

// Launcher for custom_softmax_lastdim
extern "C" void custom_softmax_lastdim_launcher(
    const float* input,
    float* output,
    const int64_t* in_shape,
    const int64_t* in_strides,
    int64_t ndim,
    cudaStream_t stream)
{
    int64_t outer_size = 1;
    for (int64_t i = 0; i < ndim-1; ++i) {
        outer_size *= in_shape[i];
    }
    int64_t W = in_shape[ndim-1];
    if (W <= 0 || outer_size <= 0) return;

    int threads = 256;
    dim3 grid((unsigned int)outer_size);
    dim3 block(threads);

    softmax_lastdim_kernel<<<grid, block, 0, stream>>>(
        input, output, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
        in_strides[0], in_strides[1], in_strides[2], in_strides[3],
        outer_size, W);
}

__global__ void softmax_lastdim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    int64_t str0, int64_t str1, int64_t str2, int64_t str3,
    int64_t outer_size,
    int64_t W)
{
    int64_t row = blockIdx.x;
    if (row >= outer_size) return;

    int64_t coords[3];
    int64_t tmp = row;
    int64_t b = tmp / (s1 * s2);
    tmp %= (s1 * s2);
    int64_t c = tmp / s2;
    int64_t h = tmp % s2;

    int64_t base = b * str0 + c * str1 + h * str2;
    const float* row_ptr = input + base;
    float* out_row_ptr = output + base;

    __shared__ float s_max;
    __shared__ float s_sum;

    float thread_max = -FLT_MAX;
    for (int64_t i = threadIdx.x; i < W; i += blockDim.x) {
        float val = row_ptr[i * str3];
        if (val > thread_max) thread_max = val;
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) s_max = smem[0];
    __syncthreads();
    float max_val = s_max;

    float thread_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < W; i += blockDim.x) {
        thread_sum += __expf(row_ptr[i * str3] - max_val);
    }
    smem[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) s_sum = smem[0];
    __syncthreads();
    float sum = s_sum;

    for (int64_t i = threadIdx.x; i < W; i += blockDim.x) {
        float val = __expf(row_ptr[i * str3] - max_val) / sum;
        out_row_ptr[i * str3] = val;
    }
}
