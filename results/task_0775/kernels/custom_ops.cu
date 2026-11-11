#include <cuda_runtime.h>

// Kernel: batched linear forward
// Each thread computes one output element: out[m, b, o] = sum_i x[b, i] * weight[m, o, i] + bias[m, o]
// x: [batch_size, in_features]
// weights: [num_models, out_features, in_features] (contiguous)
// biases: [num_models, out_features]
// output: [num_models, batch_size, out_features]
__global__ void batched_linear_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_models
) {
    int total_elements = num_models * batch_size * out_features;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        int o = linear_idx % out_features;
        int tmp = linear_idx / out_features;
        int b = tmp % batch_size;
        int m = tmp / batch_size;

        const float* weight = weights + ((long long)m * out_features + o) * in_features;
        const float* x_row = x + (long long)b * in_features;

        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += x_row[i] * weight[i];
        }
        sum += biases[m * out_features + o];

        output[linear_idx] = sum;
    }
}

// Launcher (C interface, no PyTorch dependencies)
extern "C" void batched_linear_forward_launcher(
    const float* x,
    const float* weights,
    const float* biases,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int num_models,
    cudaStream_t stream
) {
    if (batch_size <= 0 || in_features <= 0 || out_features <= 0 || num_models <= 0) {
        return; // nothing to do
    }

    int total_elements = num_models * batch_size * out_features;
    int threads = 256;
    // Cap grid to a reasonable maximum; grid-stride loop handles the rest
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) {
        blocks = 65535;
    }

    batched_linear_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, weights, biases, output,
        batch_size, in_features, out_features, num_models
    );
}
