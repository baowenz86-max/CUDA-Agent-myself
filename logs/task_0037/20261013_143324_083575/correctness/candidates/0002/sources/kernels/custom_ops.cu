#include <cuda_runtime.h>
#include <cstdint>

// Kernel to convert int64 labels to one-hot float32 on CUDA.
// Input: int64 labels on device, output: float one-hot tensor.
__global__ void one_hot_float_kernel(const int64_t* __restrict__ input,
                                     float* __restrict__ output,
                                     int64_t total_elements,
                                     int num_classes) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int64_t label = input[idx / num_classes];
    int64_t cls = idx % num_classes;
    output[idx] = (label == cls) ? 1.0f : 0.0f;
}

// Kernel for int64 one-hot (debug).
__global__ void one_hot_int64_kernel(const int64_t* __restrict__ input,
                                     int64_t* __restrict__ output,
                                     int64_t total_elements,
                                     int num_classes) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int64_t label = input[idx / num_classes];
    int64_t cls = idx % num_classes;
    output[idx] = (label == cls) ? 1 : 0;
}

// Kernel to cast int64 to float32 (debug).
__global__ void cast_int64_to_float_kernel(const int64_t* __restrict__ input,
                                           float* __restrict__ output,
                                           int64_t total_elements) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    output[idx] = (float)input[idx];
}

// Kernel for constant padding on height and width dims.
__global__ void pad_constant_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int64_t total_output,
                                    int N, int H_in, int W_in, int C,
                                    int pad_left, int pad_right,
                                    int pad_top, int pad_bottom,
                                    float value) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= total_output) return;

    int H_out = H_in + pad_top + pad_bottom;
    int W_out = W_in + pad_left + pad_right;
    int64_t tmp = idx;
    int c = tmp % C;
    tmp /= C;
    int w_out = tmp % W_out;
    tmp /= W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int n = tmp;

    int h_in = h_out - pad_top;
    int w_in = w_out - pad_left;
    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
        int64_t in_idx = (((int64_t)n * H_in + h_in) * W_in + w_in) * C + c;
        output[idx] = input[in_idx];
    } else {
        output[idx] = value;
    }
}

// Kernel for softmax along last dimension (C).
__global__ void softmax_last_dim_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int64_t rows,
                                        int C) {
    int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* in_row = input + row * C;
    float* out_row = output + row * C;

    float max_val = in_row[0];
    for (int i = 1; i < C; ++i) {
        if (in_row[i] > max_val) max_val = in_row[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        float e = expf(in_row[i] - max_val);
        out_row[i] = e;
        sum += e;
    }
    for (int i = 0; i < C; ++i) {
        out_row[i] /= sum;
    }
}

// Launchers with C linkage.

// Host-to-device launcher for CPU input (original)
extern "C" void one_hot_float_launcher(const int64_t* input_cpu,
                                        int64_t N,
                                        int64_t H,
                                        int64_t W,
                                        int num_classes,
                                        float* output_cuda,
                                        cudaStream_t stream) {
    int64_t total = N * H * W * num_classes;
    if (total == 0) return;

    int64_t* input_dev;
    cudaMalloc(&input_dev, N * H * W * sizeof(int64_t));
    cudaMemcpyAsync(input_dev, input_cpu, N * H * W * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    one_hot_float_kernel<<<blocks, threads, 0, stream>>>(input_dev, output_cuda, total, num_classes);

    cudaFree(input_dev);
}

// Device-to-device launcher for CUDA input
extern "C" void one_hot_float_cuda_launcher(const int64_t* input_dev,
                                             int64_t N,
                                             int64_t H,
                                             int64_t W,
                                             int num_classes,
                                             float* output_cuda,
                                             cudaStream_t stream) {
    int64_t total = N * H * W * num_classes;
    if (total == 0) return;

    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    one_hot_float_kernel<<<blocks, threads, 0, stream>>>(input_dev, output_cuda, total, num_classes);
}

// Host-to-device launcher for int64 one-hot (debug)
extern "C" void one_hot_int64_launcher(const int64_t* input_cpu,
                                        int64_t N,
                                        int64_t H,
                                        int64_t W,
                                        int num_classes,
                                        int64_t* output_cuda,
                                        cudaStream_t stream) {
    int64_t total = N * H * W * num_classes;
    if (total == 0) return;

    int64_t* input_dev;
    cudaMalloc(&input_dev, N * H * W * sizeof(int64_t));
    cudaMemcpyAsync(input_dev, input_cpu, N * H * W * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    one_hot_int64_kernel<<<blocks, threads, 0, stream>>>(input_dev, output_cuda, total, num_classes);

    cudaFree(input_dev);
}

// Device-to-device launcher for int64 one-hot (debug)
extern "C" void one_hot_int64_cuda_launcher(const int64_t* input_dev,
                                             int64_t N,
                                             int64_t H,
                                             int64_t W,
                                             int num_classes,
                                             int64_t* output_cuda,
                                             cudaStream_t stream) {
    int64_t total = N * H * W * num_classes;
    if (total == 0) return;

    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    one_hot_int64_kernel<<<blocks, threads, 0, stream>>>(input_dev, output_cuda, total, num_classes);
}

extern "C" void cast_float32_launcher(const int64_t* input_cuda,
                                       float* output_cuda,
                                       int64_t total_elements,
                                       cudaStream_t stream) {
    if (total_elements == 0) return;
    int threads = 256;
    int64_t blocks = (total_elements + threads - 1) / threads;
    cast_int64_to_float_kernel<<<blocks, threads, 0, stream>>>(input_cuda, output_cuda, total_elements);
}

extern "C" void pad_constant_launcher(const float* input_cuda,
                                      float* output_cuda,
                                      int64_t total_output,
                                      int N, int H_in, int W_in, int C,
                                      int pad_left, int pad_right,
                                      int pad_top, int pad_bottom,
                                      float value,
                                      cudaStream_t stream) {
    if (total_output == 0) return;
    int threads = 256;
    int64_t blocks = (total_output + threads - 1) / threads;
    pad_constant_kernel<<<blocks, threads, 0, stream>>>(
        input_cuda, output_cuda, total_output,
        N, H_in, W_in, C,
        pad_left, pad_right, pad_top, pad_bottom, value);
}

extern "C" void softmax_dim_launcher(const float* input_cuda,
                                     float* output_cuda,
                                     int64_t rows,
                                     int C,
                                     cudaStream_t stream) {
    if (rows == 0) return;
    int threads = 256;
    int64_t blocks = (rows + threads - 1) / threads;
    softmax_last_dim_kernel<<<blocks, threads, 0, stream>>>(input_cuda, output_cuda, rows, C);
}
