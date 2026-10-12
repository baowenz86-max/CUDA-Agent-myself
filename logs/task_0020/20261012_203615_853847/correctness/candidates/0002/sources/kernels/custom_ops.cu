#include <cuda_runtime.h>

// ConvTranspose1d forward kernel
// Computes output[b][oc][j] = bias[oc] + sum_{ic, k} input[b][ic][i] * weight[ic][oc][k]
// where i = (j + padding - k) / stride if divisible and 0 <= i < input_length.
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding
) {
    int total = batch_size * out_channels * output_length;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_total = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total; idx += stride_total) {
        int b = idx / (out_channels * output_length);
        int rem = idx % (out_channels * output_length);
        int oc = rem / output_length;
        int j = rem % output_length;

        float sum = bias[oc];
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                int i = (j + padding - k);
                if (i < 0 || i % stride != 0) continue;
                i /= stride;
                if (i >= 0 && i < input_length) {
                    sum += input[(b * in_channels + ic) * input_length + i] *
                           weight[(ic * out_channels + oc) * kernel_size + k];
                }
            }
        }
        output[idx] = sum;
    }
}

// Sign kernel: out[i] = (in[i] > 0) ? 1 : ((in[i] < 0) ? -1 : 0)
__global__ void sign_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total; idx += stride) {
        float val = input[idx];
        if (val > 0.0f) {
            output[idx] = 1.0f;
        } else if (val < 0.0f) {
            output[idx] = -1.0f;
        } else {
            output[idx] = 0.0f;
        }
    }
}

// Embedding kernel: output[b][j][d] = weight[index[b][j]][d]
__global__ void embedding_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int embedding_dim,
    int num_embeddings
) {
    int total = batch_size * seq_len * embedding_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total; idx += stride) {
        int d = idx % embedding_dim;
        int j = (idx / embedding_dim) % seq_len;
        int b = idx / (embedding_dim * seq_len);
        int index = (int)indices[b * seq_len + j];
        // Assuming indices are within [0, num_embeddings), no bounds check needed if trusted.
        output[idx] = weight[index * embedding_dim + d];
    }
}

// Launcher for ConvTranspose1d
// Parameters: input, weight, bias are float32 tensors on CUDA.
// output is preallocated with shape [batch_size, out_channels, output_length].
void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream
) {
    int total = batch_size * out_channels * output_length;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Limit blocks to avoid excessive grid size; grid-stride loop handles rest.
    if (blocks > 65535) blocks = 65535;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels, input_length, output_length,
        kernel_size, stride, padding
    );
}

// Launcher for Sign
void sign_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    sign_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}

// Launcher for Embedding
void embedding_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int batch_size,
    int seq_len,
    int embedding_dim,
    int num_embeddings,
    cudaStream_t stream
) {
    int total = batch_size * seq_len * embedding_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    embedding_kernel<<<blocks, threads, 0, stream>>>(
        indices, weight, output,
        batch_size, seq_len, embedding_dim, num_embeddings
    );
}

// C-interface launchers with extern "C" to avoid mangling issues.
extern "C" {
    void conv_transpose1d_forward(
        const float* input,
        const float* weight,
        const float* bias,
        float* output,
        int batch_size,
        int in_channels,
        int out_channels,
        int input_length,
        int output_length,
        int kernel_size,
        int stride,
        int padding,
        cudaStream_t stream
    ) {
        conv_transpose1d_launcher(input, weight, bias, output,
                                  batch_size, in_channels, out_channels,
                                  input_length, output_length, kernel_size, stride, padding, stream);
    }

    void sign_forward(
        const float* input,
        float* output,
        int total,
        cudaStream_t stream
    ) {
        sign_launcher(input, output, total, stream);
    }

    void embedding_forward(
        const int64_t* indices,
        const float* weight,
        float* output,
        int batch_size,
        int seq_len,
        int embedding_dim,
        int num_embeddings,
        cudaStream_t stream
    ) {
        embedding_launcher(indices, weight, output,
                           batch_size, seq_len, embedding_dim, num_embeddings, stream);
    }
}
