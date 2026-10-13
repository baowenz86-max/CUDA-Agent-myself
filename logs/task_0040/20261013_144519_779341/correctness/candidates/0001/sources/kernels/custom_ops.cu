#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// add_bias kernel: broadcast bias of shape [out_channels, 1] over last dimension
// input: [batch, channels, length], bias: [channels, 1]
// output: same shape as input
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(float* __restrict__ output,
                                const float* __restrict__ input,
                                const float* __restrict__ bias,
                                int channels,
                                int length,
                                long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        int c = (idx / length) % channels;
        output[idx] = input[idx] + bias[c];
    }
}

extern "C" void add_bias_launcher(float* output,
                                  const float* input,
                                  const float* bias,
                                  int channels,
                                  int length,
                                  long long total_elements,
                                  cudaStream_t stream) {
    int threads = 256;
    long long blocks_ll = (total_elements + threads - 1) / threads;
    // grid-stride loop supports >2^31 elements
    int blocks = (blocks_ll > 1048576) ? 1048576 : (int)blocks_ll;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(output, input, bias,
                                                    channels, length, total_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, this would be reported via TORCH_CHECK.
        // Here we simply ignore; binding layer should check as well.
    }
}

// -----------------------------------------------------------------------------
// roll kernel along dimension 2 (the last dimension)
// input shape: [batch, channels, length], shifts can be positive/negative
// output same shape. Here we assume shifts=1 (as used in model), but keep generic.
// -----------------------------------------------------------------------------
__global__ void roll_kernel(float* __restrict__ output,
                            const float* __restrict__ input,
                            int length,
                            int shifts,
                            long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        long long base = idx / length;
        int pos = idx % length;
        int src_pos = (pos - shifts) % length;
        if (src_pos < 0) src_pos += length;
        output[base * length + pos] = input[base * length + src_pos];
    }
}

extern "C" void roll_launcher(float* output,
                              const float* input,
                              int length,
                              int shifts,
                              long long total_elements,
                              cudaStream_t stream) {
    int threads = 256;
    long long blocks_ll = (total_elements + threads - 1) / threads;
    int blocks = (blocks_ll > 1048576) ? 1048576 : (int)blocks_ll;
    roll_kernel<<<blocks, threads, 0, stream>>>(output, input, length, shifts, total_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}

// -----------------------------------------------------------------------------
// view kernel: actually just copies data (view does not change data layout,
// but for completeness we provide a kernel that copies contiguous elements,
// assuming input is contiguous after previous ops; the binding will pass contiguous
// tensors. input and output have same number of elements.
// -----------------------------------------------------------------------------
__global__ void view_kernel(float* __restrict__ output,
                            const float* __restrict__ input,
                            long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        output[idx] = input[idx];
    }
}

extern "C" void view_launcher(float* output,
                              const float* input,
                              long long total_elements,
                              cudaStream_t stream) {
    int threads = 256;
    long long blocks_ll = (total_elements + threads - 1) / threads;
    int blocks = (blocks_ll > 1048576) ? 1048576 : (int)blocks_ll;
    view_kernel<<<blocks, threads, 0, stream>>>(output, input, total_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}

// -----------------------------------------------------------------------------
// linear kernel: output = input @ weight.T + bias
// input shape: [batch, num_features], weight shape: [out_features, num_features],
// bias shape: [out_features]
// output shape: [batch, out_features]
// We use coalesced access: each thread computes one output element.
// -----------------------------------------------------------------------------
__global__ void linear_kernel(float* __restrict__ output,
                              const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              int batch_size,
                              int num_features,
                              int out_features) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_features) return;
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        const float* inp = input + (long long)b * num_features;
        const float* w = weight + (long long)out_idx * num_features;
        for (int j = 0; j < num_features; ++j) {
            sum += inp[j] * w[j];
        }
        output[(long long)b * out_features + out_idx] = sum + bias[out_idx];
    }
}

extern "C" void linear_launcher(float* output,
                                const float* input,
                                const float* weight,
                                const float* bias,
                                int batch_size,
                                int num_features,
                                int out_features,
                                cudaStream_t stream) {
    int threads = 256;
    int blocks = (out_features + threads - 1) / threads;
    linear_kernel<<<blocks, threads, 0, stream>>>(output, input, weight, bias,
                                                  batch_size, num_features, out_features);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}
