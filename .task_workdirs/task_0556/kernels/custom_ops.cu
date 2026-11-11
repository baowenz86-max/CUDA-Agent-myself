#include <cuda_runtime.h>

__global__ void embedding_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t embedding_dim)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t pos = idx; pos < total_elements; pos += stride) {
        int64_t token = indices[pos];
        const float* src_row = weight + token * embedding_dim;
        float* dst_row = output + pos * embedding_dim;
        for (int64_t d = 0; d < embedding_dim; ++d) {
            dst_row[d] = src_row[d];
        }
    }
}

template <int BLOCK_DIM>
__global__ void layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t rows,
    int64_t dim,
    float eps,
    bool elementwise_affine)
{
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * dim;
    float* row_out = output + row * dim;

    __shared__ float s_mean;
    __shared__ float s_var;
    __shared__ float s_weight[BLOCK_DIM];
    __shared__ float s_bias[BLOCK_DIM];

    if (elementwise_affine) {
        if (weight != nullptr) {
            for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
                s_weight[i] = weight[i];
            }
        } else {
            for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
                s_weight[i] = 1.0f;
            }
        }
        if (bias != nullptr) {
            for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
                s_bias[i] = bias[i];
            }
        } else {
            for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
                s_bias[i] = 0.0f;
            }
        }
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
        float v = row_in[i];
        local_sum += v;
        local_sq_sum += v * v;
    }

    __shared__ float s_sum[BLOCK_DIM / 32];
    __shared__ float s_sq_sum[BLOCK_DIM / 32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
    }
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sq_sum[warp_id] = local_sq_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        float total_sq_sum = 0.0f;
        int num_warps = BLOCK_DIM / 32;
        for (int i = 0; i < num_warps; ++i) {
            total_sum += s_sum[i];
            total_sq_sum += s_sq_sum[i];
        }
        float mean = total_sum / dim;
        float var = total_sq_sum / dim - mean * mean;
        s_mean = mean;
        s_var = var;
    }
    __syncthreads();

    float mean = s_mean;
    float var = s_var;
    float inv_std = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
        float normalized = (row_in[i] - mean) * inv_std;
        if (elementwise_affine) {
            float w = s_weight[i];
            float b = s_bias[i];
            row_out[i] = normalized * w + b;
        } else {
            row_out[i] = normalized;
        }
    }
}

extern "C" void custom_embedding_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t batch,
    int64_t seq_len,
    int64_t vocab_size,
    int64_t embedding_dim,
    bool indices_on_device,
    bool weight_on_device,
    cudaStream_t stream)
{
    int64_t total_positions = batch * seq_len;
    int64_t* d_indices = nullptr;
    float* d_weight = nullptr;

    if (indices_on_device) {
        d_indices = const_cast<int64_t*>(indices);
    } else {
        cudaError_t err = cudaMalloc(&d_indices, total_positions * sizeof(int64_t));
        if (err != cudaSuccess) return;
        cudaMemcpyAsync(d_indices, indices, total_positions * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    }

    if (weight_on_device) {
        d_weight = const_cast<float*>(weight);
    } else {
        cudaError_t err = cudaMalloc(&d_weight, vocab_size * embedding_dim * sizeof(float));
        if (err != cudaSuccess) {
            if (!indices_on_device) cudaFree(d_indices);
            return;
        }
        cudaMemcpyAsync(d_weight, weight, vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
    }

    int block_size = 256;
    int grid_size = (int)((total_positions + block_size - 1) / block_size);
    embedding_kernel<<<grid_size, block_size, 0, stream>>>(d_indices, d_weight, output, total_positions, embedding_dim);

    // If we allocated temporary device memory for host pointers, free after kernel completion.
    if (!indices_on_device && d_indices) {
        cudaFree(d_indices);
    }
    if (!weight_on_device && d_weight) {
        cudaFree(d_weight);
    }
}

extern "C" void custom_layer_norm_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t rows,
    int64_t dim,
    double eps,
    bool elementwise_affine,
    bool weight_on_device,
    bool bias_on_device,
    cudaStream_t stream)
{
    float* d_weight = nullptr;
    float* d_bias = nullptr;

    if (elementwise_affine) {
        if (weight != nullptr) {
            if (weight_on_device) {
                d_weight = const_cast<float*>(weight);
            } else {
                cudaError_t err = cudaMalloc(&d_weight, dim * sizeof(float));
                if (err != cudaSuccess) return;
                cudaMemcpyAsync(d_weight, weight, dim * sizeof(float), cudaMemcpyHostToDevice, stream);
            }
        }
        if (bias != nullptr) {
            if (bias_on_device) {
                d_bias = const_cast<float*>(bias);
            } else {
                cudaError_t err = cudaMalloc(&d_bias, dim * sizeof(float));
                if (err != cudaSuccess) {
                    if (d_weight && !weight_on_device) cudaFree(d_weight);
                    return;
                }
                cudaMemcpyAsync(d_bias, bias, dim * sizeof(float), cudaMemcpyHostToDevice, stream);
            }
        }
    }

    int block_dim = 32;
    while (block_dim < dim) block_dim <<= 1;
    if (block_dim > 1024) block_dim = 1024;

    dim3 grid(rows);
    switch (block_dim) {
        case 32:
            layer_norm_kernel<32><<<grid, 32, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        case 64:
            layer_norm_kernel<64><<<grid, 64, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        case 128:
            layer_norm_kernel<128><<<grid, 128, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        case 256:
            layer_norm_kernel<256><<<grid, 256, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        case 512:
            layer_norm_kernel<512><<<grid, 512, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        case 1024:
            layer_norm_kernel<1024><<<grid, 1024, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        default:
            layer_norm_kernel<32><<<grid, 32, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
    }

    if (d_weight && !weight_on_device) cudaFree(d_weight);
    if (d_bias && !bias_on_device) cudaFree(d_bias);
}
