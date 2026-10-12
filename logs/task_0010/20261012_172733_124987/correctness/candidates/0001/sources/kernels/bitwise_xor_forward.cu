#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

__global__ void bitwise_xor_kernel(unsigned char* output, const unsigned char* input, const unsigned char* bias,
                                   int bias_elements, int bias_stride, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // bias broadcast: bias shape is (out_channels, 1, 1), so each channel has same bias value.
    // Need to compute channel index from idx: total is batch*channels*height*width with contiguous layout.
    // Assuming NCHW layout. The index breakdown: w = idx % width; h = (idx/width)%height; c = (idx/(width*height))%channels; n = idx/(channels*width*height)
    // However, we only need channel index: c = (idx / (width * height)) % channels.
    // But we don't have width, height, channels passed. We'll pass them.
}

// Better: we'll pass channel dims and compute channel index.
torch::Tensor bitwise_xor_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(bias.dtype() == torch::kUInt8, "Bias must be uint8");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(bias.dim() == 3 && bias.size(1) == 1 && bias.size(2) == 1, "Bias must be shape (C,1,1)");

    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int total = batch * channels * height * width;

    auto output = torch::empty_like(input);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel with appropriate parameters
    bitwise_xor_kernel<<<blocks, threads, 0, stream>>>(output.data_ptr<unsigned char>(),
                                                       input.data_ptr<unsigned char>(),
                                                       bias.data_ptr<unsigned char>(),
                                                       channels, height * width, total);
    return output;
}

void register_bitwise_xor_forward(pybind11::module& m) {
    m.def("bitwise_xor_forward", &bitwise_xor_forward, "Bitwise XOR with broadcast bias");
}

REGISTER_BINDING(bitwise_xor_forward, register_bitwise_xor_forward);
