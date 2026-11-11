// CUDA kernels for unfold, amax over dim=(1,2), and count_nonzero over dim=(1,2)
#include <cuda_runtime.h>
#include <cstdint>

// Helper function to compute output spatial dimensions for unfold
__host__ __device__ inline int unfold_output_size(int input_size, int kernel_size, int dilation, int padding, int stride) {
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

// Kernel for unfold_forward: writes output tensor of shape [N, C*kh*kw, L]
// Each thread computes one output element
template <typename T>
__global__ void unfold_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int padding_h, int padding_w,
    int stride_h, int stride_w,
    int out_h, int out_w,
    int L, int Ckhw)  // C * kernel_h * kernel_w
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * Ckhw * L;
    if (idx >= total) return;

    // Decompose idx into n, ck, l
    int l = idx % L;
    int ck = (idx / L) % Ckhw;
    int n = idx / (L * Ckhw);

    // Decompose ck into c, ky, kx (C, kernel_h, kernel_w order)
    int kx = ck % kernel_w;
    int ky = (ck / kernel_w) % kernel_h;
    int c = ck / (kernel_w * kernel_h);

    // Spatial position in unfold output
    int ow = l % out_w;
    int oh = l / out_w;

    // Input pixel corresponding to this output (with padding)
    int ih = oh * stride_h - padding_h + ky * dilation_h;
    int iw = ow * stride_w - padding_w + kx * dilation_w;

    T val = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? input[((int64_t)n * C + c) * H * W + (int64_t)ih * W + iw] : (T)0;
    output[idx] = val;
}

// Kernel for unfold_amax_forward: computes max over dim=(1,2) for each batch
// Each block handles one batch, using parallel reduction over Ckhw * L elements
template <typename T>
__global__ void unfold_amax_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int padding_h, int padding_w,
    int stride_h, int stride_w,
    int out_h, int out_w,
    int L, int Ckhw)
{
    extern __shared__ T sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (bid >= N) return;

    // Total elements per batch: Ckhw * L
    int64_t total_elems = (int64_t)Ckhw * L;
    T local_max = (T)(-1e30); // negative infinity approximation for float/double
    // Use grid-stride within block over all elements
    for (int64_t i = tid; i < total_elems; i += blockDim.x) {
        int l = i % L;
        int ck = i / L;
        // decompose ck
        int kx = ck % kernel_w;
        int ky = (ck / kernel_w) % kernel_h;
        int c = ck / (kernel_w * kernel_h);
        int ow = l % out_w;
        int oh = l / out_w;
        int ih = oh * stride_h - padding_h + ky * dilation_h;
        int iw = ow * stride_w - padding_w + kx * dilation_w;
        T val = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? input[((int64_t)bid * C + c) * H * W + (int64_t)ih * W + iw] : (T)0;
        if (val > local_max) local_max = val;
    }

    // Block reduction in shared memory
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            T other = sdata[tid + s];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[bid] = sdata[0];
    }
}

// Kernel for unfold_count_nonzero_forward: counts non-zero over dim=(1,2)
// Each block handles one batch, using atomic add to global output (or we can use one block per batch)
template <typename T>
__global__ void unfold_count_nonzero_kernel(
    const T* __restrict__ input,
    int64_t* __restrict__ output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int padding_h, int padding_w,
    int stride_h, int stride_w,
    int out_h, int out_w,
    int L, int Ckhw)
{
    extern __shared__ int64_t sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (bid >= N) return;

    int64_t total_elems = (int64_t)Ckhw * L;
    int64_t local_count = 0;
    for (int64_t i = tid; i < total_elems; i += blockDim.x) {
        int l = i % L;
        int ck = i / L;
        int kx = ck % kernel_w;
        int ky = (ck / kernel_w) % kernel_h;
        int c = ck / (kernel_w * kernel_h);
        int ow = l % out_w;
        int oh = l / out_w;
        int ih = oh * stride_h - padding_h + ky * dilation_h;
        int iw = ow * stride_w - padding_w + kx * dilation_w;
        T val = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? input[((int64_t)bid * C + c) * H * W + (int64_t)ih * W + iw] : (T)0;
        if (val != (T)0) {
            ++local_count;
        }
    }

    // Block reduction (sum)
    sdata[tid] = local_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[bid] = sdata[0];
    }
}

// Launcher functions (extern "C" so binding.cpp can call them)
// All launchers assume input is contiguous, on CUDA, float32, and shapes are valid.
// They allocate output via torch::empty in binding, but here just take output pointers.

extern "C" void unfold_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_h, int64_t kernel_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t padding_h, int64_t padding_w,
    cudaStream_t stream)
{
    int stride_h = 1;
    int stride_w = 1;
    int out_h = unfold_output_size((int)H, (int)kernel_h, (int)dilation_h, (int)padding_h, stride_h);
    int out_w = unfold_output_size((int)W, (int)kernel_w, (int)dilation_w, (int)padding_w, stride_w);
    int L = out_h * out_w;
    int Ckhw = (int)C * (int)kernel_h * (int)kernel_w;

    int64_t total = (int64_t)N * Ckhw * L;
    const int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    int blocks = (blocks64 > 0) ? (int)blocks64 : 1;

    unfold_forward_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output,
        (int)N, (int)C, (int)H, (int)W,
        (int)kernel_h, (int)kernel_w,
        (int)dilation_h, (int)dilation_w,
        (int)padding_h, (int)padding_w,
        stride_h, stride_w,
        out_h, out_w, L, Ckhw);
}

extern "C" void unfold_amax_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_h, int64_t kernel_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t padding_h, int64_t padding_w,
    cudaStream_t stream)
{
    int stride_h = 1;
    int stride_w = 1;
    int out_h = unfold_output_size((int)H, (int)kernel_h, (int)dilation_h, (int)padding_h, stride_h);
    int out_w = unfold_output_size((int)W, (int)kernel_w, (int)dilation_w, (int)padding_w, stride_w);
    int L = out_h * out_w;
    int Ckhw = (int)C * (int)kernel_h * (int)kernel_w;

    const int threads = 256;  // can tune, must be power of two
    int blocks = (int)N;
    size_t shared_mem = threads * sizeof(float);

    unfold_amax_kernel<float><<<blocks, threads, shared_mem, stream>>>(
        input, output,
        (int)N, (int)C, (int)H, (int)W,
        (int)kernel_h, (int)kernel_w,
        (int)dilation_h, (int)dilation_w,
        (int)padding_h, (int)padding_w,
        stride_h, stride_w,
        out_h, out_w, L, Ckhw);
}

extern "C" void unfold_count_nonzero_launcher(
    const float* input,
    int64_t* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_h, int64_t kernel_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t padding_h, int64_t padding_w,
    cudaStream_t stream)
{
    int stride_h = 1;
    int stride_w = 1;
    int out_h = unfold_output_size((int)H, (int)kernel_h, (int)dilation_h, (int)padding_h, stride_h);
    int out_w = unfold_output_size((int)W, (int)kernel_w, (int)dilation_w, (int)padding_w, stride_w);
    int L = out_h * out_w;
    int Ckhw = (int)C * (int)kernel_h * (int)kernel_w;

    const int threads = 256;
    int blocks = (int)N;
    size_t shared_mem = threads * sizeof(int64_t);

    unfold_count_nonzero_kernel<float><<<blocks, threads, shared_mem, stream>>>(
        input, output,
        (int)N, (int)C, (int)H, (int)W,
        (int)kernel_h, (int)kernel_w,
        (int)dilation_h, (int)dilation_w,
        (int)padding_h, (int)padding_w,
        stride_h, stride_w,
        out_h, out_w, L, Ckhw);
}
