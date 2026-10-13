#include <cuda_runtime.h>
#include <float.h>

// ReLU forward kernel: elementwise max(x, 0)
__global__ void relu_forward_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t total_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

// Softmax forward kernel: softmax over last dimension
// Each block handles one row (outer_size rows), each thread processes one element
// blockDim.x = 256, inner_size = 224, so one thread per element, some threads inactive
__global__ void softmax_forward_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int64_t outer_size,
                                       int64_t inner_size) {
    int64_t row = blockIdx.x;
    if (row >= outer_size) return;

    const float* row_input = input + row * inner_size;
    float* row_output = output + row * inner_size;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    // Compute max over row
    float max_val = -FLT_MAX;
    if (tid < inner_size) {
        max_val = row_input[tid];
    }
    // Block reduction for max (simple tree reduction using shared memory)
    __shared__ float s_max[256];
    s_max[tid] = max_val;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    max_val = s_max[0];

    // Compute sum of exp(x - max)
    float sum = 0.0f;
    if (tid < inner_size) {
        sum = __expf(row_input[tid] - max_val);
    }
    // Block reduction for sum
    __shared__ float s_sum[256];
    s_sum[tid] = sum;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    sum = s_sum[0];

    // Normalize and write output
    if (tid < inner_size) {
        row_output[tid] = __expf(row_input[tid] - max_val) / sum;
    }
}

// Launcher for ReLU
void relu_forward_launcher(const float* input,
                           float* output,
                           int64_t total_elements,
                           cudaStream_t stream) {
    if (total_elements <= 0) return;
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    relu_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

// Launcher for Softmax
void softmax_forward_launcher(const float* input,
                              float* output,
                              int64_t outer_size,
                              int64_t inner_size,
                              cudaStream_t stream) {
    if (outer_size <= 0 || inner_size <= 0) return;
    int threads = 256;  // must be >= inner_size for this kernel (224 <= 256)
    softmax_forward_kernel<<<(int)outer_size, threads, 0, stream>>>(
        input, output, outer_size, inner_size);
}
