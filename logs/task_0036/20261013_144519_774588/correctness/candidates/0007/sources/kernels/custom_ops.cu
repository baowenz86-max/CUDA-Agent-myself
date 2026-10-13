// kernels/custom_ops.cu
// CUDA kernels and launchers for add, argmin_dim, and stft_complex
// No torch/ATen/pybind dependencies.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------
// add: elementwise addition x + y, float32
// ---------------------------------------------------------------------
__global__ void add_kernel(float* out, const float* x, const float* y, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        out[i] = x[i] + y[i];
    }
}

extern "C" void add_launcher(float* out, const float* x, const float* y, int64_t n, cudaStream_t stream) {
    const int block = 256;
    int64_t grid64 = (n + block - 1) / block;
    int grid = (int)std::min<int64_t>(grid64, 65535);
    add_kernel<<<grid, block, 0, stream>>>(out, x, y, n);
}

// ---------------------------------------------------------------------
// argmin along last dimension, keepdim=True (output shape [batch,1])
// Input x: [batch, length] float32, row-major or strided
// Output: int64 indices [batch,1]
// Each row handled by one block to keep simple and exact tie-breaking.
// ---------------------------------------------------------------------
__global__ void argmin_kernel(const float* x, int64_t* out, int64_t length, int64_t x_stride0) {
    int64_t row = blockIdx.x;
    const float* row_ptr = x + row * x_stride0;
    int64_t best_idx = 0;
    float best_val = row_ptr[0];
    for (int64_t i = 1; i < length; ++i) {
        float val = row_ptr[i];
        if (val < best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    if (threadIdx.x == 0) {
        out[row] = best_idx;
    }
}

extern "C" void argmin_launcher(const float* x, int64_t* out, int64_t batch, int64_t length, int64_t x_stride0, cudaStream_t stream) {
    // One block per row
    argmin_kernel<<<(int)batch, 256, 0, stream>>>(x, out, length, x_stride0);
}

// ---------------------------------------------------------------------
// STFT with rectangular window, center=True, pad_mode='reflect',
// one-sided, return_complex=True. Uses cuFFT R2C.
// Input x: [batch, length] float32 (assumed contiguous in this task)
// Output: complex64 [batch, n_freq, n_frames], where n_freq = n_fft/2+1,
// n_frames = 1 + (length + 2*pad - n_fft)/hop_length, pad = n_fft/2.
// ---------------------------------------------------------------------
extern "C" void stft_launcher(
    const float* x,           // input [batch, length] contiguous
    cufftComplex* out,        // output [batch, n_freq, n_frames] complex64
    int64_t batch,
    int64_t length,
    int n_fft,
    int hop_length,
    int win_length,          // ignored in computation, only for API compatibility (window is rectangular)
    cudaStream_t stream
) {
    const int pad = n_fft / 2;
    const int n_freq = n_fft / 2 + 1;
    const int n_frames = 1 + (int)((length + 2 * pad - n_fft) / hop_length);

    // Allocate workspace for padded frame (n_fft floats) and cuFFT output (n_freq complex)
    float* d_frame;
    cufftComplex* d_fft_out;
    cudaMalloc(&d_frame, sizeof(float) * n_fft);
    cudaMalloc(&d_fft_out, sizeof(cufftComplex) * n_freq);

    // Create cuFFT plan for single R2C
    cufftHandle plan;
    cufftResult res = cufftPlan1d(&plan, n_fft, CUFFT_R2C, 1); // batch size 1, we loop over frames
    if (res != CUFFT_SUCCESS) {
        // Handle error (we can't throw; binding will set error if we return early, but we proceed)
        // For simplicity, return; binding should check after launch but we cannot report error here.
        cudaFree(d_frame);
        cudaFree(d_fft_out);
        return;
    }
    cufftSetStream(plan, stream);

    // Kernel to fill reflected padded frame and execute FFT? Since we need cuFFT per frame, we cannot
    // launch FFT from device. We'll launch reflection kernel to copy data into d_frame, then call cufftExecR2C,
    // then copy result to output. This is done serially per frame in host loop, which may be slow.
    // To improve, we could batch all frames with many plans, but simpler host loop for correctness.
    // However, the loop over 128*782=100k frames with host launches may be very slow, but acceptable for baseline?
    // We'll implement straightforwardly.

    // We'll use a kernel to fill d_frame with reflect padding from input row, then call cufftExecR2C,
    // then copy result to appropriate location in output.
    // Define kernel for reflection fill
    // We'll declare it as a global function later.
    
    // Use a single frame buffer, process all frames sequentially on stream.
    // Since we need to synchronize after each FFT to reuse buffer, this serializes, but acceptable.
    for (int64_t b = 0; b < batch; ++b) {
        const float* x_row = x + b * length;
        // Output row: out + b * (n_freq * n_frames), with each frame contiguous in frequency dimension
        // But contract says stride [401166, 1, 513] for output, which is unusual. Actually shape [128,513,782]
        // and stride [401166,1,513] means the last dimension (frames) has stride 513? Let's check: For tensor
        // shape [128,513,782], contiguous stride would be [513*782, 782, 1] = [401166, 782, 1]. But contract
        // says stride [401166, 1, 513]. That means the tensor layout is not standard: dimension 1 (freq) has
        // stride 1, dimension 2 (frames) has stride 513. This suggests the actual output from torch.stft with
        // return_complex=True is not contiguous? Wait, PyTorch's stft returns a tensor with shape
        // [batch, n_fft//2+1, n_frames] and strides [n_freq*n_frames, 1, n_freq]. That is exactly
        // [401166, 1, 513]. So the frame dimension is the last, but the memory layout has frequency as the
        // contiguous dimension and frames spaced by n_freq. So output[b, f, t] is at b*stride0 + f*stride1 + t*stride2
        // = b*401166 + f*1 + t*513.
        // So for each frame t, the n_freq complex values are contiguous in memory, and next frame starts after n_freq.
        // This matches our out pointer linear layout: out[b*n_freq*n_frames + t*n_freq + f].
        // So we can treat output as contiguous array of complex values with shape [batch, n_frames, n_freq] in memory.
        cufftComplex* out_row = out + b * n_freq * n_frames;

        for (int frame = 0; frame < n_frames; ++frame) {
            int start = frame * hop_length - pad;
            // Fill frame with reflect padding from x_row of length length
            // Use a kernel to fill d_frame
            // Kernel: fill_reflect_frame_kernel<<<...>>>(d_frame, x_row, length, start, n_fft)
            // But need to define kernel.
            // We'll define below.
            // For now, call kernel (assuming defined).
            int threads = 256;
            int blocks = (n_fft + threads - 1) / threads;
            fill_reflect_frame_kernel<<<blocks, threads, 0, stream>>>(d_frame, x_row, length, start, n_fft);

            // Execute cuFFT R2C on d_frame -> d_fft_out
            cufftExecR2C(plan, d_frame, d_fft_out);

            // Copy n_freq complex values to out_row + frame * n_freq
            // Use cudaMemcpyAsync within stream? Could use cudaMemcpyAsync D2D, but it's synchronous in stream order.
            cudaMemcpyAsync(out_row + frame * n_freq, d_fft_out, sizeof(cufftComplex) * n_freq, cudaMemcpyDeviceToDevice, stream);
        }
    }

    cufftDestroy(plan);
    cudaFree(d_frame);
    cudaFree(d_fft_out);
}

// Reflect padding kernel: fills d_frame (length n_fft) with values from x_row (length),
// using reflect padding around boundaries. start is the index in the padded coordinate
// such that frame[i] corresponds to (start+i) in padded sequence of length length+2*pad.
// Reflect mapping: for index j in padded sequence [-pad, length+pad-1],
// actual index = reflect(j, length), where reflect is PyTorch's reflect mode.
__global__ void fill_reflect_frame_kernel(float* frame, const float* x_row, int64_t length, int start, int n_fft) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_fft) return;
    int64_t pos = start + idx; // position in padded coordinates
    // reflect index
    int64_t src;
    // Use PyTorch reflect: reflect index i over [0, length-1] as
    // if length == 1: always 0
    if (length == 1) {
        src = 0;
    } else {
        // periodic with period 2*(length-1)
        int64_t period = 2 * (length - 1);
        // adjust pos to non-negative modulo period
        int64_t m = pos % period;
        if (m < 0) m += period;
        if (m < length) {
            src = m;
        } else {
            src = period - m;
        }
        // ensure in range [0, length-1]
        if (src >= length) src = length - 1;
    }
    frame[idx] = x_row[src];
}
