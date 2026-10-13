// custom_ops.cu - CUDA kernels and launchers for add, argmin, STFT
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>

// -----------------------------------------------------------------------------
// Add: output = x + y elementwise
// -----------------------------------------------------------------------------
template <typename T>
__global__ void add_kernel(const T* __restrict__ x,
                           const T* __restrict__ y,
                           T* __restrict__ out,
                           long long n) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        out[idx] = x[idx] + y[idx];
    }
}

extern "C" cudaError_t add_forward_launcher(const float* x,
                                            const float* y,
                                            float* out,
                                            long long n,
                                            cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    const int threads = 256;
    int blocks = (int)std::min<long long>((n + threads - 1) / threads, 1048576);
    add_kernel<float><<<blocks, threads, 0, stream>>>(x, y, out, n);
    cudaError_t err = cudaGetLastError();
    return err;
}

// -----------------------------------------------------------------------------
// Argmin over last dimension, keepdim=True. Input [B, N] -> output [B, 1] int64.
// Redesigned to avoid shared memory corruption.
// Each block processes one row (B blocks). Within a block, each thread handles a
// subset of N elements and computes a local min. Then a reduction in shared memory
// with correct offsets and synchronization yields the global index.
// -----------------------------------------------------------------------------
__global__ void argmin_lastdim_kernel(const float* __restrict__ x,
                                      long long* __restrict__ out,
                                      int N,
                                      long long B) {
    long long b = blockIdx.x;
    if (b >= B) return;
    const float* row = x + b * (long long)N;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Shared memory: use a union to ensure proper alignment and avoid overlap.
    // We'll store pairs of (int index, float value).
    extern __shared__ char smem[];
    int* s_indices = reinterpret_cast<int*>(smem);
    float* s_values = reinterpret_cast<float*>(smem + blockDim.x * sizeof(int));

    int best_idx = 0;
    float best_val = row[0]; // Initialize with first element for each thread; but we want each thread to start with its own first element, not row[0] for all.
    // Actually, each thread should initialize with its first assigned element to avoid bias.
    if (tid < N) {
        best_idx = tid;
        best_val = row[tid];
    } else {
        best_val = INFINITY;
        best_idx = -1;
    }

    // Stride over elements assigned to this thread
    for (int i = tid + stride; i < N; i += stride) {
        float val = row[i];
        if (val < best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    // Store local results in shared memory
    s_indices[tid] = best_idx;
    s_values[tid] = best_val;
    __syncthreads();

    // Reduction over blockDim.x elements (which may be > N if N < blockDim.x, but we handle with -1 sentinel)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_val = s_values[tid + s];
            int other_idx = s_indices[tid + s];
            // If either is invalid (-1 sentinel for idx), treat as +inf
            float my_val = s_values[tid];
            int my_idx = s_indices[tid];
            if (other_idx != -1 && (my_idx == -1 || other_val < my_val)) {
                s_values[tid] = other_val;
                s_indices[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (s_indices[0] != -1) {
            out[b] = (long long)s_indices[0];
        } else {
            // This should not happen as N>0, but set 0 to avoid garbage.
            out[b] = 0;
        }
    }
}

extern "C" cudaError_t argmin_launcher(const float* x,
                                       long long* out,
                                       int N,
                                       long long B,
                                       cudaStream_t stream) {
    if (B == 0 || N == 0) return cudaSuccess;
    int threads = std::min(1024, N);
    // Ensure threads is power of two for reduction (up to 1024)
    // If N not power of two, we still use max threads 1024, reduction works for any power-of-2 <= blockDim.
    // Actually, if N < threads, some threads have no elements and best_idx=-1, which we handle.
    // But we still need blockDim power-of-2 for tree reduction. We'll use 1024 always if N>1024 else next power of two? For simplicity, use 1024 if N>=1024 else N (which may not be power of two causing reduction issue).
    // Better: use threads = min(1024, next_power_of_two(N)). But N may be 200000 > 1024, so threads=1024. If N smaller, we can use threads = N if N is power of two else next power of two > N, capped at 1024.
    // For correctness with any N, we can use threads = 256 or 512 etc. Let's compute next power of two for N <=1024.
    if (N < 1024) {
        int power = 1;
        while (power < N) power <<= 1;
        threads = power;
    } else {
        threads = 1024;
    }
    int shared_bytes = threads * sizeof(int) + threads * sizeof(float);
    dim3 grid((unsigned int)std::min<long long>(B, 1048576));
    argmin_lastdim_kernel<<<grid, threads, shared_bytes, stream>>>(x, out, N, B);
    cudaError_t err = cudaGetLastError();
    return err;
}

// -----------------------------------------------------------------------------
// STFT with cuFFT, center=True, reflect padding, Hann window, onesided output.
// Input: [B, N] float32, output: [B, F, T] complex64 (interleaved float2).
// F = n_fft/2 + 1, T = 1 + floor(N / hop_length).
// -----------------------------------------------------------------------------
__global__ void stft_build_frames_kernel(const float* __restrict__ x,
                                         int B,
                                         int N,
                                         int n_fft,
                                         int hop_length,
                                         int win_length,
                                         const float* __restrict__ window,
                                         cufftComplex* __restrict__ frames,
                                         int T) {
    int t = blockIdx.x;
    int b = blockIdx.y;
    if (t >= T || b >= B) return;
    int L = n_fft;
    int start = t * hop_length - n_fft / 2;
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        int idx = start + i;
        float val = 0.0f;
        if (idx >= 0 && idx < N) {
            val = x[b * N + idx];
        } else {
            int reflected = idx;
            if (reflected < 0) reflected = -reflected - 1;
            else if (reflected >= N) reflected = 2 * N - 1 - reflected;
            if (reflected >= 0 && reflected < N) {
                val = x[b * N + reflected];
            }
        }
        frames[(b * T + t) * L + i].x = val * window[i];
        frames[(b * T + t) * L + i].y = 0.0f;
    }
}

__global__ void stft_extract_onesided_kernel(const cufftComplex* __restrict__ frames,
                                             cufftComplex* __restrict__ out,
                                             int B,
                                             int T,
                                             int L,
                                             int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (b >= B) return;
    int total = F * T;
    if (idx >= total) return;
    int f = idx % F;
    int t = idx / F;
    out[(b * F + f) * T + t] = frames[(b * T + t) * L + f];
}

extern "C" cudaError_t stft_forward_launcher(const float* x,
                                            int B,
                                            int N,
                                            int n_fft,
                                            int hop_length,
                                            int win_length,
                                            cufftComplex* out,
                                            cudaStream_t stream) {
    if (B == 0 || N == 0) return cudaSuccess;
    const int F = n_fft / 2 + 1;
    const int T = 1 + N / hop_length;
    const int L = n_fft;

    cufftHandle plan;
    cufftResult result = cufftCreate(&plan);
    if (result != CUFFT_SUCCESS) return cudaErrorUnknown;
    result = cufftSetStream(plan, stream);
    if (result != CUFFT_SUCCESS) { cufftDestroy(plan); return cudaErrorUnknown; }
    result = cufftPlan1d(&plan, n_fft, CUFFT_C2C, B * T);
    if (result != CUFFT_SUCCESS) { cufftDestroy(plan); return cudaErrorUnknown; }

    cufftComplex* d_frames;
    cudaError_t err = cudaMallocAsync(&d_frames, (size_t)B * T * L * sizeof(cufftComplex), stream);
    if (err != cudaSuccess) { cufftDestroy(plan); return err; }

    float* h_window = new float[n_fft]();
    int left = (n_fft - win_length) / 2;
    for (int i = 0; i < win_length; ++i) {
        h_window[left + i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (win_length - 1)));
    }
    float* d_window;
    err = cudaMallocAsync(&d_window, n_fft * sizeof(float), stream);
    if (err != cudaSuccess) { delete[] h_window; cudaFreeAsync(d_frames, stream); cufftDestroy(plan); return err; }
    err = cudaMemcpyAsync(d_window, h_window, n_fft * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        delete[] h_window;
        cudaFreeAsync(d_window, stream);
        cudaFreeAsync(d_frames, stream);
        cufftDestroy(plan);
        return err;
    }
    delete[] h_window;

    dim3 blockBuild(256);
    dim3 gridBuild(T, B);
    stft_build_frames_kernel<<<gridBuild, blockBuild, 0, stream>>>(
        x, B, N, n_fft, hop_length, win_length, d_window, d_frames, T);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFreeAsync(d_window, stream);
        cudaFreeAsync(d_frames, stream);
        cufftDestroy(plan);
        return err;
    }

    result = cufftExecC2C(plan, d_frames, d_frames, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) {
        cudaFreeAsync(d_window, stream);
        cudaFreeAsync(d_frames, stream);
        cufftDestroy(plan);
        return cudaErrorUnknown;
    }

    dim3 blockExtract(256);
    dim3 gridExtract((F * T + blockExtract.x - 1) / blockExtract.x, B);
    stft_extract_onesided_kernel<<<gridExtract, blockExtract, 0, stream>>>(
        d_frames, out, B, T, L, F);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFreeAsync(d_window, stream);
        cudaFreeAsync(d_frames, stream);
        cufftDestroy(plan);
        return err;
    }

    cudaFreeAsync(d_window, stream);
    cudaFreeAsync(d_frames, stream);
    cufftDestroy(plan);

    return cudaSuccess;
}
