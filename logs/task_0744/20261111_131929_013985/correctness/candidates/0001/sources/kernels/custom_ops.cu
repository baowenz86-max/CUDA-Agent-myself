// kernels/custom_ops.cu
// CUDA kernels and launchers for mean_dim0_keepdim and batchnorm2d_eval
// No PyTorch dependencies; launchers use raw pointers, scalar sizes, and cudaStream_t.

#include <cuda_runtime.h>

// Kernel: compute mean over dimension 0 with keepdim=True
// input shape: [batch, channels, height, width]
// output shape: [1, channels, height, width]
// Supports arbitrary strides via input_stride_batch, input_stride_channel, input_stride_height, input_stride_width.
// Each thread computes one output element (channel c, height h, width w) by summing over batch and dividing by batch.
__global__ void mean_dim0_keepdim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int height,
    int width,
    long long input_stride_batch,
    long long input_stride_channel,
    long long input_stride_height,
    long long input_stride_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = channels * height * width;
    if (idx >= total_elements) return;

    // Decode idx into (c, h, w)
    int c = idx / (height * width);
    int rem = idx - c * height * width;
    int h = rem / width;
    int w = rem - h * width;

    // Accumulate over batch dimension
    float sum = 0.0f;
    const float* input_base = input + c * input_stride_channel + h * input_stride_height + w * input_stride_width;
    for (int b = 0; b < batch; ++b) {
        sum += input_base[b * input_stride_batch];
    }

    output[idx] = sum / (float)batch;
}

// Kernel: evaluation-mode BatchNorm2d
// input shape: [N, C, H, W] (typically N=1 after mean reduction, but supports N>=1)
// weight, bias, running_mean, running_var are per-channel arrays of length C.
// weight and bias may be null; if null, scale=1 and bias=0.
// Runs per element: y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * scale + bias
// Supports arbitrary strides for input/output (assumed same strides for input and output, since output shape matches input).
// For simplicity, this kernel handles contiguous or strided tensors by using provided strides.
__global__ void batchnorm2d_eval_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W,
    long long input_stride_N,
    long long input_stride_C,
    long long input_stride_H,
    long long input_stride_W,
    long long output_stride_N,
    long long output_stride_C,
    long long output_stride_H,
    long long output_stride_W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    if (idx >= total_elements) return;

    // Decode flat idx into (n, c, h, w)
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = rsqrtf(var + eps);
    float scale = (weight != nullptr) ? weight[c] : 1.0f;
    float shift = (bias != nullptr) ? bias[c] : 0.0f;

    float x = input[n * input_stride_N + c * input_stride_C + h * input_stride_H + w * input_stride_W];
    float y = (x - mean) * inv_std * scale + shift;

    output[n * output_stride_N + c * output_stride_C + h * output_stride_H + w * output_stride_W] = y;
}

// Launcher for mean_dim0_keepdim
// Expects data on current CUDA stream (passed in).
// input_ptr: pointer to input tensor data (float)
// output_ptr: pointer to output tensor data (float), allocated by caller
// batch, channels, height, width: dimensions
// strides: as long long (int64_t) for large tensors
// stream: CUDA stream
void mean_dim0_keepdim_launcher(
    const float* input_ptr,
    float* output_ptr,
    int batch,
    int channels,
    int height,
    int width,
    long long input_stride_batch,
    long long input_stride_channel,
    long long input_stride_height,
    long long input_stride_width,
    cudaStream_t stream)
{
    int total_elements = channels * height * width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Grid-stride loop not strictly needed because each thread handles one output, but we ensured blocks cover all.
    // We also could add grid-stride loop if blocks > 65535, but total_elements here is ~6M, blocks ~24k, within limit.
    // However for safety we might ensure grid size within limits. 24k < 2^31-1, fine.
    // We can optionally clamp blocks to max grid size and use grid-stride loop, but not necessary.
    mean_dim0_keepdim_kernel<<<blocks, threads, 0, stream>>>(
        input_ptr, output_ptr,
        batch, channels, height, width,
        input_stride_batch, input_stride_channel, input_stride_height, input_stride_width);
}

// Launcher for batchnorm2d_eval
// weight_ptr and bias_ptr may be nullptr to indicate no affine parameters.
void batchnorm2d_eval_launcher(
    const float* input_ptr,
    float* output_ptr,
    const float* weight_ptr,
    const float* bias_ptr,
    const float* running_mean_ptr,
    const float* running_var_ptr,
    float eps,
    int N, int C, int H, int W,
    long long input_stride_N,
    long long input_stride_C,
    long long input_stride_H,
    long long input_stride_W,
    long long output_stride_N,
    long long output_stride_C,
    long long output_stride_H,
    long long output_stride_W,
    cudaStream_t stream)
{
    int total_elements = N * C * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // For very large tensors, blocks might exceed limit; but for our shapes, total ~6M, blocks ~24k, safe.
    // Add grid-stride loop inside kernel if we ever need to handle >2^31 elements.
    batchnorm2d_eval_kernel<<<blocks, threads, 0, stream>>>(
        input_ptr, output_ptr,
        weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
        eps,
        N, C, H, W,
        input_stride_N, input_stride_C, input_stride_H, input_stride_W,
        output_stride_N, output_stride_C, output_stride_H, output_stride_W);
}

extern "C" {
    // Exported C-style launcher for mean_dim0_keepdim (used by binding)
    void mean_dim0_keepdim_cuda(
        const float* input_ptr,
        float* output_ptr,
        int64_t batch,
        int64_t channels,
        int64_t height,
        int64_t width,
        int64_t input_stride_batch,
        int64_t input_stride_channel,
        int64_t input_stride_height,
        int64_t input_stride_width,
        cudaStream_t stream)
    {
        // Cast int64_t to int for kernel launch; check range
        if (batch > 2147483647 || channels > 2147483647 || height > 2147483647 || width > 2147483647) {
            // Should not happen; ignore or handle error (but no exception here)
            return;
        }
        mean_dim0_keepdim_launcher(
            input_ptr, output_ptr,
            (int)batch, (int)channels, (int)height, (int)width,
            (long long)input_stride_batch, (long long)input_stride_channel,
            (long long)input_stride_height, (long long)input_stride_width,
            stream);
    }

    // Exported C-style launcher for batchnorm2d_eval
    void batchnorm2d_eval_cuda(
        const float* input_ptr,
        float* output_ptr,
        const float* weight_ptr,
        const float* bias_ptr,
        const float* running_mean_ptr,
        const float* running_var_ptr,
        float eps,
        int64_t N,
        int64_t C,
        int64_t H,
        int64_t W,
        int64_t input_stride_N,
        int64_t input_stride_C,
        int64_t input_stride_H,
        int64_t input_stride_W,
        int64_t output_stride_N,
        int64_t output_stride_C,
        int64_t output_stride_H,
        int64_t output_stride_W,
        cudaStream_t stream)
    {
        if (N > 2147483647 || C > 2147483647 || H > 2147483647 || W > 2147483647) {
            return;
        }
        batchnorm2d_eval_launcher(
            input_ptr, output_ptr,
            weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
            eps,
            (int)N, (int)C, (int)H, (int)W,
            (long long)input_stride_N, (long long)input_stride_C,
            (long long)input_stride_H, (long long)input_stride_W,
            (long long)output_stride_N, (long long)output_stride_C,
            (long long)output_stride_H, (long long)output_stride_W,
            stream);
    }
}
