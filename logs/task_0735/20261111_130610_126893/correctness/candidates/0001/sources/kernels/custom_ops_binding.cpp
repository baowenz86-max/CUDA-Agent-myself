#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" cudaError_t custom_elu_launcher(const float* input,
                                            float* output,
                                            long long n,
                                            double alpha,
                                            cudaStream_t stream);

extern "C" cudaError_t custom_relu_launcher(const float* input,
                                             float* output,
                                             long long n,
                                             cudaStream_t stream);

torch::Tensor custom_elu(torch::Tensor input, double alpha) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    // Ensure contiguous tensor for launcher (launcher assumes contiguous data)
    auto input_contig = input.contiguous();
    
    auto output = torch::empty_like(input_contig);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    auto status = custom_elu_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        input_contig.numel(),
        alpha,
        stream
    );
    TORCH_CHECK(status == cudaSuccess, "custom_elu_launcher failed with error: ", cudaGetErrorString(status));
    
    return output;
}

torch::Tensor custom_relu(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    auto input_contig = input.contiguous();
    
    auto output = torch::empty_like(input_contig);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    auto status = custom_relu_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        input_contig.numel(),
        stream
    );
    TORCH_CHECK(status == cudaSuccess, "custom_relu_launcher failed with error: ", cudaGetErrorString(status));
    
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_elu", &custom_elu, "Custom ELU forward",
          py::arg("input"), py::arg("alpha"));
    m.def("custom_relu", &custom_relu, "Custom ReLU forward",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
