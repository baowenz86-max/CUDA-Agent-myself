#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// ------------------------- Pad Constant 2D -------------------------
__global__ void pad_constant_2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int channels, int in_h, int in_w,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    float pad_value) {
    int out_h = in_h + pad_top + pad_bottom;
    int out_w = in_w + pad_left + pad_right;
    int64_t total = (int64_t)batch * channels * out_h * out_w;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int w = idx % out_w;
        int tmp = idx / out_w;
        int h = tmp % out_h;
        tmp /= out_h;
        int c = tmp % channels;
        int b = tmp / channels;
        int in_h_idx = h - pad_top;
        int in_w_idx = w - pad_left;
        if (in_h_idx >= 0 && in_h_idx < in_h && in_w_idx >= 0 && in_w_idx < in_w) {
            output[idx] = input[((b * channels + c) * in_h + in_h_idx) * in_w + in_w_idx];
        } else {
            output[idx] = pad_value;
        }
    }
}

extern "C" void pad_constant_2d_launcher(
    const float* input, float* output,
    int batch, int channels, int in_h, int in_w,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    float pad_value, cudaStream_t stream) {
    int out_h = in_h + pad_top + pad_bottom;
    int out_w = in_w + pad_left + pad_right;
    int64_t total = (int64_t)batch * channels * out_h * out_w;
    int threads = 256;
    int blocks = (int)std::min((int64_t)((total + threads - 1) / threads), (int64_t)1048576);
    if (blocks == 0) return;
    pad_constant_2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, in_h, in_w,
        pad_top, pad_bottom, pad_left, pad_right, pad_value);
}

torch::Tensor pad_constant_2d(torch::Tensor input, std::vector<int64_t> padding, double value) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(padding.size() == 4, "padding must have 4 elements");
    int pad_left = padding[0];
    int pad_right = padding[1];
    int pad_top = padding[2];
    int pad_bottom = padding[3];
    auto sizes = input.sizes();
    int64_t out_h = sizes[2] + pad_top + pad_bottom;
    int64_t out_w = sizes[3] + pad_left + pad_right;
    auto output = torch::empty({sizes[0], sizes[1], out_h, out_w}, input.options());
    if (output.numel() == 0) return output;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    pad_constant_2d_launcher(
        input.data_ptr<float>(), output.data_ptr<float>(),
        input.size(0), input.size(1), input.size(2), input.size(3),
        pad_top, pad_bottom, pad_left, pad_right,
        (float)value, stream);
    return output;
}

// ------------------------- Rot90 k=1 dims=(2,3) -------------------------
__global__ void rot90_k1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * height * width;
    if (idx >= total) return;
    int w = idx % width;
    int tmp = idx / width;
    int h = tmp % height;
    tmp /= height;
    int c = tmp % channels;
    int b = tmp / channels;
    int new_h = width;
    int new_w = height;
    // rot90 with k=1: new[i][j] = old[j][new_w-1-i]? Actually for dims (2,3) and k=1:
    // output[b,c,i,j] = input[b,c, j, height-1-i]
    int out_i = w;
    int out_j = height - 1 - h;
    int out_idx = ((b * channels + c) * new_h + out_i) * new_w + out_j;
    output[out_idx] = input[idx];
}

extern "C" void rot90_k1_launcher(
    const float* input, float* output,
    int batch, int channels, int height, int width,
    cudaStream_t stream) {
    int total = batch * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rot90_k1_kernel<<<blocks, threads, 0, stream>>>(input, output, batch, channels, height, width);
}

torch::Tensor rot90_k1(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32 && input.is_contiguous(), "invalid input");
    TORCH_CHECK(input.dim() == 4, "expected 4D");
    auto sizes = input.sizes();
    int64_t out_h = sizes[3];
    int64_t out_w = sizes[2];
    auto output = torch::empty({sizes[0], sizes[1], out_h, out_w}, input.options());
    if (output.numel() == 0) return output;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    rot90_k1_launcher(input.data_ptr<float>(), output.data_ptr<float>(),
                      input.size(0), input.size(1), input.size(2), input.size(3), stream);
    return output;
}

// ------------------------- Conv2d forward -------------------------
// Tiled convolution with shared memory for input tile and constant weight caching
// Input: [N, C, H, W], Weight: [K, C, R, S], Bias: [K]
// Output: [N, K, OH, OW]
// stride=1, dilation=1, groups=1 (as per reference)

template<int TILE_H, int TILE_W, int BLOCK_H, int BLOCK_W>
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int OH, int OW) {
    __shared__ float tile_input[(TILE_H + R - 1) * (TILE_W + S - 1)];
    __shared__ float tile_weight[K * C * R * S]; // constant for all blocks across batch?

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z; // batch index

    int out_row = by * BLOCK_H + ty;
    int out_col = bx * BLOCK_W + tx;

    if (bz >= N) return;

    // Load weight into shared memory once per block (coalesced)
    int total_weight = K * C * R * S;
    if (tx + ty * BLOCK_W < total_weight) {
        int wid = tx + ty * BLOCK_W;
        if (wid < total_weight) {
            tile_weight[wid] = weight[wid];
        }
    }
    __syncthreads();

    if (out_row < OH && out_col < OW) {
        float accum = 0.0f;
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    int in_row = out_row + r;
                    int in_col = out_col + s;
                    // input index: ((bz*C + c)*H + in_row)*W + in_col
                    float in_val = input[((bz * C + c) * H + in_row) * W + in_col];
                    // weight index: ((k*C + c)*R + r)*S + s
                    float w_val = tile_weight[((0 * C + c) * R + r) * S + s]; // k=0 placeholder, will need loop over K
                    // Actually we need to compute for all K output channels for this output position.
                    // We will loop over K inside this thread.
                    // So better restructure: each thread computes one output element for a given K, so K should be in blockIdx.z or threadIdx.z.
                    // Let's simplify: use 3D block with z for K? But K may be large. We can use 1D grid over K.
                }
            }
        }
    }
}

// Simpler implementation: each thread computes one output element (n,k,oh,ow)
// grid: (OW, OH, K*N) or (OW, OH, K, N) using 3D grid with z = n*K+k
__global__ void conv2d_direct_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int OH, int OW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nk = blockIdx.z; // combined n and k
    if (ow >= OW || oh >= OH) return;
    int n = nk / K;
    int k = nk % K;
    if (n >= N) return;
    float sum = bias[k];
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int in_row = oh + r;
                int in_col = ow + s;
                float in_val = input[((n * C + c) * H + in_row) * W + in_col];
                float w_val = weight[((k * C + c) * R + r) * S + s];
                sum += in_val * w_val;
            }
        }
    }
    int out_idx = ((n * K + k) * OH + oh) * OW + ow;
    output[out_idx] = sum;
}

extern "C" void conv2d_forward_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C, int H, int W, int K, int R, int S,
    int OH, int OW, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((OW + block.x - 1) / block.x, (OH + block.y - 1) / block.y, N * K);
    conv2d_direct_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output, N, C, H, W, K, R, S, OH, OW);
}

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "must be float32");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(), "contiguous required");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "expected 4D tensors");
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0);
    int R = weight.size(2);
    int S = weight.size(3);
    TORCH_CHECK(weight.size(1) == C, "input channels mismatch");
    TORCH_CHECK(bias.numel() == K, "bias size mismatch");
    int OH = H - R + 1;
    int OW = W - S + 1;
    auto output = torch::empty({N, K, OH, OW}, input.options());
    if (output.numel() == 0) return output;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv2d_forward_launcher(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H, W, K, R, S, OH, OW, stream);
    return output;
}

// ------------------------- IsInf -------------------------
__global__ void isinf_kernel(const float* __restrict__ input, bool* __restrict__ output, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        output[idx] = isinf(input[idx]);
    }
}

extern "C" void isinf_launcher(const float* input, bool* output, int64_t total, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)std::min((int64_t)((total + threads - 1) / threads), (int64_t)1048576);
    if (blocks == 0) return;
    isinf_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}

torch::Tensor isinf(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32 && input.is_contiguous(), "invalid input");
    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));
    if (output.numel() == 0) return output;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    isinf_launcher(input.data_ptr<float>(), output.data_ptr<bool>(), input.numel(), stream);
    return output;
}

// ------------------------- Masked Fill Zero -------------------------
__global__ void masked_fill_zero_kernel(float* __restrict__ data, const bool* __restrict__ mask, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        if (mask[idx]) {
            data[idx] = 0.0f;
        }
    }
}

extern "C" void masked_fill_zero_launcher(float* data, const bool* mask, int64_t total, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)std::min((int64_t)((total + threads - 1) / threads), (int64_t)1048576);
    if (blocks == 0) return;
    masked_fill_zero_kernel<<<blocks, threads, 0, stream>>>(data, mask, total);
}

torch::Tensor masked_fill_zero(torch::Tensor input, torch::Tensor mask) {
    TORCH_CHECK(input.is_cuda() && mask.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && mask.dtype() == torch::kBool, "dtype mismatch");
    TORCH_CHECK(input.sizes() == mask.sizes(), "size mismatch");
    TORCH_CHECK(input.is_contiguous() && mask.is_contiguous(), "contiguous required");
    // Operate in-place on input to avoid extra allocation
    if (input.numel() == 0) return input;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    masked_fill_zero_launcher(input.data_ptr<float>(), mask.data_ptr<bool>(), input.numel(), stream);
    return input;
}

// ------------------------- Add Bias -------------------------
// bias shape: [K, 1, 1] broadcast over [N, K, H, W]
__global__ void add_bias_kernel(float* __restrict__ data, const float* __restrict__ bias, int K, int H, int W, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        // compute w, h, k, n from idx
        int w = idx % W;
        int64_t tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int k = tmp % K;
        data[idx] += bias[k];
    }
}

extern "C" void add_bias_launcher(float* data, const float* bias, int K, int H, int W, int64_t total, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)std::min((int64_t)((total + threads - 1) / threads), (int64_t)1048576);
    if (blocks == 0) return;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(data, bias, K, H, W, total);
}

torch::Tensor add_bias(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "dtype mismatch");
    TORCH_CHECK(input.dim() == 4 && bias.dim() == 3, "dim mismatch");
    TORCH_CHECK(bias.size(0) == input.size(1), "bias channel mismatch");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias must be [K,1,1]");
    TORCH_CHECK(input.is_contiguous() && bias.is_contiguous(), "contiguous required");
    // Operate in-place on input
    if (input.numel() == 0) return input;
    int K = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(input.data_ptr<float>(), bias.data_ptr<float>(), K, H, W, input.numel(), stream);
    return input;
}

// ------------------------- Register all bindings -------------------------
void register_ops(pybind11::module& m) {
    m.def("pad_constant_2d", &pad_constant_2d, "ConstantPad2d");
    m.def("rot90_k1", &rot90_k1, "Rot90 with k=1 on dims 2,3");
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward");
    m.def("isinf", &isinf, "Elementwise isinf");
    m.def("masked_fill_zero", &masked_fill_zero, "Masked fill with zero");
    m.def("add_bias", &add_bias, "Add bias with broadcasting");
}

REGISTER_BINDING(ops, register_ops);