#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void aap1d_forward_launcher(const float* input,
                                       float* output,
                                       int64_t batch_size,
                                       int64_t input_size,
                                       int64_t seq_length,
                                       int64_t output_size,
                                       cudaStream_t stream);

extern "C" void masked_select_forward_launcher(const float* input,
                                               float* output,
                                               int64_t total_elements,
                                               cudaStream_t stream);

// Adaptive average pooling 1D forward
at::Tensor aap1d_forward(at::Tensor input, int64_t output_size) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, C, L)");
    TORCH_CHECK(output_size > 0, "output_size must be positive");

    auto batch_size = input.size(0);
    auto input_size_feat = input.size(1);  // avoid name clash with variable
    auto seq_length = input.size(2);

    auto output = at::empty({batch_size, input_size_feat, output_size}, input.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    aap1d_forward_launcher(input.data_ptr<float>(),
                           output.data_ptr<float>(),
                           batch_size,
                           input_size_feat,
                           seq_length,
                           output_size,
                           stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "aap1d_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Masked select forward for scalar mask
at::Tensor masked_select_forward(at::Tensor input, at::Tensor mask) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(mask.scalar_type() == at::kBool, "mask must be bool");
    TORCH_CHECK(mask.numel() == 1, "mask must be a scalar tensor");

    // Read scalar mask value from device to host
    bool mask_val;
    cudaMemcpyAsync(&mask_val, mask.data_ptr<bool>(), sizeof(bool), cudaMemcpyDeviceToHost,
                    at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

    if (!mask_val) {
        return at::empty({0}, input.options());
    }

    auto output = at::empty({input.numel()}, input.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    masked_select_forward_launcher(input.data_ptr<float>(),
                                   output.data_ptr<float>(),
                                   input.numel(),
                                   stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "masked_select_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("aap1d_forward", &aap1d_forward, "Adaptive average pooling 1D forward",
          py::arg("input"), py::arg("output_size"));
    m.def("masked_select_forward", &masked_select_forward, "Masked select forward (scalar mask)",
          py::arg("input"), py::arg("mask"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
