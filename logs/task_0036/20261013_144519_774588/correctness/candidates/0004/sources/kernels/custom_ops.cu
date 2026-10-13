// custom_ops.cu - CUDA kernels and launchers for custom_add, custom_argmin, custom_stft
// No torch/ATen/c10/pybind dependencies. Pure CUDA and cuFFT.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>
#include <algorithm>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            abort(); \
        } \
    } while (0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
            abort(); \
        } \
    } while (0)

// ---------------------------------------------------------------------
// 1. custom_add: element-wise addition
// ---------------------------------------------------------------------

__global__ void elementwise_add_kernel(const float* __restrict__ x,
                                       const float* __restrict__ y,
                                       float* __restrict__ out,
                                       int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        out[i] = x[i] + y[i];
    }
}

extern "C" void custom_add_launcher(const float* x,
                                    const float* y,
                                    float* out,
                                    int64_t n,
                                    cudaStream_t stream) {
    const int threads = 256;
    int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 65535);
    if (blocks <= 0) blocks = 1;
    elementwise_add_kernel<<<blocks, threads, 0, stream>>>(x, y, out, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------
// 2. custom_argmin: argmin along last dimension, keepdim=True
// ---------------------------------------------------------------------

__global__ void argmin_kernel(const float* __restrict__ x,
                              int64_t* __restrict__ out,
                              int64_t rows,
                              int64_t cols) {
    // Each block handles one row, returns global index of minimum
    extern __shared__ float sdata[];
    extern __shared__ int64_t sidx[];
    // We need both sdata and sidx; actually can use one shared array of struct, but here we use separate extern pointers.
    // CUDA requires dynamic shared memory to be declared as one array. So we'll use a single char array and cast in kernel.
    // Actually extern __shared__ float sdata[]; and then sidx[] would share same memory incorrectly.
    // So we need to do union approach: use a struct or simply allocate shared memory of size 2 * blockDim.x * (sizeof(float)+sizeof(int64_t)).
    // But easier: we'll not use shared memory reduction; each thread linearly scans a portion and then a single warp reduces? 
    // For simplicity and correctness (though slower), each block does reduction using shared memory for both value and index.
    // We'll use a struct Pair { float val; int64_t idx; } in shared memory.
    struct Pair { float val; int64_t idx; };
    __shared__ Pair spair[1024]; // max threads 1024 (we use 256)
    int tid = threadIdx.x;
    int64_t row = blockIdx.x;
    const float* row_ptr = x + row * cols;
    float min_val = FLT_MAX;
    int64_t min_idx = -1;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        float v = row_ptr[col];
        if (v < min_val) {
            min_val = v;
            min_idx = col;
        }
    }
    spair[tid].val = min_val;
    spair[tid].idx = min_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (spair[tid + s].val < spair[tid].val) {
                spair[tid] = spair[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[row] = spair[0].idx;
    }
}

extern "C" void custom_argmin_launcher(const float* x,
                                       int64_t* out,
                                       int64_t rows,
                                       int64_t cols,
                                       cudaStream_t stream) {
    int threads = 256;
    if (cols < 256) threads = (int)cols;
    int shared_bytes = threads * sizeof(float) + threads * sizeof(int64_t);
    // We asssume threads <= 1024, shared memory fits.
    argmin_kernel<<<rows, threads, shared_bytes, stream>>>(x, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------
// 3. custom_stft: STFT using cuFFT R2C
// ---------------------------------------------------------------------

// Device function: reflect padding index beyond boundaries
__device__ int reflect_index(int idx, int len) {
    if (idx < 0) idx = -idx;
    if (idx >= len) {
        int period = 2 * (len - 1);
        idx = idx % period;
        if (idx >= len) idx = period - idx;
    }
    return idx;
}

extern "C" void custom_stft_launcher(const float* x,
                                     int n_fft,
                                     int hop_length,
                                     int win_length,
                                     bool return_complex,
                                     float* out_real,
                                     float* out_imag,
                                     int64_t B,
                                     int64_t L,
                                     cudaStream_t stream) {
    // Compute number of frames: center=True with reflect pad adds n_fft//2 on both sides.
    int n_frames = 1 + (L + n_fft / 2 + n_fft / 2 - n_fft) / hop_length;
    // Simplify: n_frames = 1 + (L + n_fft - n_fft) / hop_length = 1 + L / hop_length, but careful with integer.
    n_frames = 1 + (L + n_fft - n_fft) / hop_length; // center=True with reflect pad effectively adds n_fft/2 each side.
    // Actually formula: n_frames = 1 + (L + n_fft/2 + n_fft/2 - n_fft) / hop_length = 1 + L / hop_length.
    // Given L=200000, hop=256 => 1 + 781 = 782, matches.
    int freq = n_fft / 2 + 1; // 513

    // Allocate temporary device buffers for padded input and frames
    int64_t padded_len = L + n_fft; // reflect pad adds n_fft/2 each side => total L + n_fft
    float* d_padded = nullptr;
    CUDA_CHECK(cudaMalloc(&d_padded, padded_len * B * sizeof(float)));
    // Allocate buffer for the windowed frames: each frame of length n_fft, total frames = n_frames per batch
    float* d_frames = nullptr;
    CUDA_CHECK(cudaMalloc(&d_frames, B * n_frames * n_fft * sizeof(float)));
    // Allocate complex output buffer for FFT: B * n_frames * freq complex floats
    cufftComplex* d_fft_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fft_out, B * n_frames * freq * sizeof(cufftComplex)));

    // Kernel to perform reflect padding and copy to d_padded
    // We can combine padding and frame construction into one kernel that writes windowed frames directly,
    // but for simplicity we first pad, then run cuFFT on frames created by a kernel from padded.
    // We'll create a kernel to pad reflect.
    dim3 pad_block(256);
    dim3 pad_grid((padded_len * B + pad_block.x - 1) / pad_block.x);
    // We'll use a custom kernel: reflect_pad_kernel
    auto reflect_pad_kernel = [](const float* __restrict__ x, float* __restrict__ padded,
                                 int64_t L, int n_fft, int64_t padded_len) __global__ {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total = padded_len * gridDim.x; // wrong, need B separate but we treat padded as B*padded_len contiguous
        // Actually we launched with grid covering total elements, so each thread handles one element.
        if (idx >= total) return;
        int64_t b = idx / padded_len;
        int64_t pos = idx % padded_len;
        int64_t orig_pos = pos - n_fft / 2;
        int orig_idx = reflect_index((int)orig_pos, (int)L);
        padded[idx] = x[b * L + orig_idx];
    };
    // We cannot define lambda as __global__, so we define a separate kernel below.
    // We'll instead write a separate __global__ function outside.

    // Launch padding kernel
    reflect_pad_kernel<<<pad_grid, pad_block, 0, stream>>>(x, d_padded, L, n_fft, padded_len);
    CUDA_CHECK(cudaGetLastError());

    // Now create frames by selecting windows from padded data and multiply by rectangular window (all ones).
    // We can write a kernel that from padded data copies each frame into d_frames.
    // But we can also use cuFFT with custom strides? cuFFT can handle batch with different strides using advanced data layout.
    // Simpler: create contiguous frames of length n_fft.
    dim3 frame_block(256);
    int total_frames = B * n_frames;
    dim3 frame_grid((total_frames + frame_block.x - 1) / frame_block.x);
    // Kernel: each thread handles one frame and copies n_fft elements? That would be inefficient.
    // Better: each thread copies multiple elements per frame? We'll just launch one thread per frame and loop n_fft inside (slow but okay for now).
    // Or use a kernel with one thread per element: for each frame, copy n_fft elements.
    // We'll define a kernel frame_copy_kernel that uses grid-stride over total elements = total_frames * n_fft.
    // We'll implement later.

    // For cuFFT: set up plan for many 1D R2C transforms of size n_fft, batch = B*n_frames.
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n_fft, CUFFT_R2C, B * n_frames));
    // Execute FFT on d_frames -> d_fft_out
    CUFFT_CHECK(cufftExecR2C(plan, d_frames, d_fft_out));
    // Synchronize stream for cufft (cuFFT uses stream from plan, need to set stream)
    CUFFT_CHECK(cufftSetStream(plan, stream));

    // After FFT, copy results to output complex array with correct strides.
    // Output array in torch is complex64 with shape [B, freq, n_frames], strides [freq*n_frames, 1, freq].
    // We are given separate real and imag pointers, but torch complex tensor is interleaved real/imag.
    // In binding we'll allocate a complex tensor and pass its data_ptr as float2? Actually torch complex64 is float2 (real, imag contiguous).
    // We will pass output as float2* to launcher, but we declared out_real and out_imag separately. We can combine into a custom struct or use float2*.
    // To avoid confusion, we'll change launcher signature to accept float2* output_complex and we can cast to cufftComplex (which is float2).
    // In the final code, we'll set output as float2*, or use cufftComplex*. We'll adjust.

    // Cleanup
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_padded));
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_fft_out));
}

// Define the actual global kernels for padding and frame copying
__global__ void reflect_pad_kernel(const float* __restrict__ x,
                                   float* __restrict__ padded,
                                   int64_t L,
                                   int n_fft,
                                   int64_t padded_len,
                                   int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t b = idx / padded_len;
    int64_t pos = idx % padded_len;
    int64_t orig_pos = pos - n_fft / 2;
    int orig_idx = reflect_index((int)orig_pos, (int)L);
    padded[idx] = x[b * L + orig_idx];
}

__global__ void frame_copy_kernel(const float* __restrict__ padded,
                                  float* __restrict__ frames,
                                  int64_t padded_len,
                                  int n_fft,
                                  int hop_length,
                                  int n_frames,
                                  int64_t total_frames,
                                  int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t frame_idx = idx / n_fft; // which frame
    int64_t offset = idx % n_fft;
    int64_t b = frame_idx / n_frames;
    int64_t f = frame_idx % n_frames;
    int64_t start = f * hop_length; // start in padded coordinates (padding already added n_fft/2)
    padded_idx = b * padded_len + start + offset;
    frames[idx] = padded[padded_idx];
}
