#include <cuda_runtime.h>
#include <device_functions.h>

// -----------------------------------------------------------------------------
// clamp_log2_forward: y = log2(clamp(x, min, +inf))
// Supports arbitrary strides via linear index and offset computation.
// Input x is float32, output y is float32, min is float.
// -----------------------------------------------------------------------------
__global__ void clamp_log2_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int64_t numel,
                                  int64_t ndim,
                                  const int64_t* __restrict__ shape,
                                  const int64_t* __restrict__ strides,
                                  float min_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // Convert linear index to multidimensional indices (row-major)
    int64_t rem = idx;
    int64_t src_offset = 0;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        int64_t coord = rem % shape[d];
        rem /= shape[d];
        src_offset += coord * strides[d];
    }

    float v = x[src_offset];
    v = fmaxf(v, min_val);
    y[idx] = log2f(v);
}

// -----------------------------------------------------------------------------
// mse_loss_forward: reduce mean((x - target)^2) -> scalar
// Two-phase: each block computes partial sum, then final reduction in a separate kernel.
// -----------------------------------------------------------------------------
__global__ void mse_loss_block_sum_kernel(const float* __restrict__ x,
                                          const float* __restrict__ target,
                                          float* __restrict__ partial_sums,
                                          int64_t numel,
                                          int64_t ndim,
                                          const int64_t* __restrict__ x_shape,
                                          const int64_t* __restrict__ x_strides,
                                          const int64_t* __restrict__ target_shape,
                                          const int64_t* __restrict__ target_strides) {
    extern __shared__ float sdata[];
    int64_t tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    // Grid-stride loop to handle arbitrary numel
    for (int64_t i = idx; i < numel; i += gridDim.x * blockDim.x) {
        // Compute source offsets for x and target (same logical shape)
        int64_t rem = i;
        int64_t x_offset = 0;
        int64_t target_offset = 0;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t coord = rem % x_shape[d];
            rem /= x_shape[d];
            x_offset += coord * x_strides[d];
            target_offset += coord * target_strides[d];
        }
        float diff = x[x_offset] - target[target_offset];
        local_sum += diff * diff;
    }

    // Block reduction
    sdata[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void mse_loss_final_reduce_kernel(const float* __restrict__ partial_sums,
                                             float* __restrict__ output,
                                             int num_blocks,
                                             int64_t numel) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0] / (float)numel;
    }
}

// -----------------------------------------------------------------------------
// ConvTranspose1d forward kernel
// Computes y[n, co, l_out] = bias[co] + sum_{ci, k} x[n, ci, l_in] * weight[ci, co, k]
// where l_in = (l_out + padding * 2 - kernel_size) / stride (integer division, if divisible)
// and k = l_out + padding - l_in * stride, with 0 <= k < kernel_size.
// Input x may have arbitrary strides, output y is contiguous.
// -----------------------------------------------------------------------------
__global__ void conv_transpose1d_kernel(const float* __restrict__ x,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias, // may be null
                                        float* __restrict__ y,
                                        int N, int C_in, int C_out, int L_in, int L_out,
                                        int kernel_size, int stride, int padding,
                                        int64_t ndim, const int64_t* __restrict__ x_shape,
                                        const int64_t* __restrict__ x_strides) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C_out * L_out;
    if (idx >= total) return;

    // Decompose idx into n, co, l_out (row-major order: N, C_out, L_out)
    int l_out = idx % L_out;
    int co = (idx / L_out) % C_out;
    int n = idx / (L_out * C_out);

    float acc = 0.0f;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int k = 0; k < kernel_size; ++k) {
            // Compute l_in such that l_out = (l_in - 1) * stride - 2*padding + k + output_padding? 
            // Actually, transposed convolution formula: output index = input_index * stride - padding + k.
            // So l_out = l_in * stride - padding + k. Hence l_in = (l_out + padding - k) / stride, if divisible.
            int t = l_out + padding - k;
            if (t % stride != 0) continue;
            int l_in = t / stride;
            if (l_in < 0 || l_in >= L_in) continue;

            // Compute offset in x for (n, ci, l_in)
            int64_t x_offset = n * x_strides[0] + ci * x_strides[1] + l_in * x_strides[2];
            float x_val = x[x_offset];

            // weight index: (ci, co, k) -> ci * (C_out * kernel_size) + co * kernel_size + k
            int w_idx = ci * (C_out * kernel_size) + co * kernel_size + k;
            acc += x_val * weight[w_idx];
        }
    }

    if (bias != nullptr) {
        acc += bias[co];
    }

    y[idx] = acc;
}

// -----------------------------------------------------------------------------
// Launchers (extern "C" for C++ binding)
// -----------------------------------------------------------------------------

// clamp_log2_forward
extern "C" void clamp_log2_launcher(const float* x,
                                    float* y,
                                    int64_t numel,
                                    int64_t ndim,
                                    const int64_t* shape,
                                    const int64_t* strides,
                                    float min_val,
                                    cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    clamp_log2_kernel<<<blocks, threads, 0, stream>>>(x, y, numel, ndim, shape, strides, min_val);
}

// mse_loss_forward
extern "C" void mse_loss_launcher(const float* x,
                                  const float* target,
                                  float* output,
                                  int64_t numel,
                                  int64_t ndim,
                                  const int64_t* x_shape,
                                  const int64_t* x_strides,
                                  const int64_t* target_shape,
                                  const int64_t* target_strides,
                                  cudaStream_t stream) {
    int threads = 256;
    int max_blocks = 1024;
    int blocks = (int)min((int64_t)max_blocks, (numel + threads - 1) / threads);
    float* partial_sums;
    cudaMalloc(&partial_sums, blocks * sizeof(float));

    size_t shared_mem = threads * sizeof(float);
    mse_loss_block_sum_kernel<<<blocks, threads, shared_mem, stream>>>(
        x, target, partial_sums, numel, ndim,
        x_shape, x_strides, target_shape, target_strides);

    int final_threads = 256;
    int final_blocks = 1;
    size_t final_shared_mem = final_threads * sizeof(float);
    mse_loss_final_reduce_kernel<<<final_blocks, final_threads, final_shared_mem, stream>>>(
        partial_sums, output, blocks, numel);

    cudaFreeAsync(partial_sums, stream);
}

// conv_transpose1d_forward launcher
extern "C" void conv_transpose1d_launcher(const float* x,
                                          const float* weight,
                                          const float* bias, // may be nullptr
                                          float* y,
                                          int N, int C_in, int C_out, int L_in, int L_out,
                                          int kernel_size, int stride, int padding,
                                          int64_t ndim, const int64_t* x_shape,
                                          const int64_t* x_strides,
                                          cudaStream_t stream) {
    int threads = 256;
    int64_t total = (int64_t)N * C_out * L_out;
    int blocks = (int)((total + threads - 1) / threads);
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, y,
        N, C_in, C_out, L_in, L_out,
        kernel_size, stride, padding,
        ndim, x_shape, x_strides);
}
