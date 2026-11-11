#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Moves input tensor to target device (always CPU for this task).
// Handles both CPU and CUDA inputs by copying data to a new CPU tensor.
torch::Tensor to_device_forward(torch::Tensor x) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty(x.sizes(), torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    if (x.device().is_cuda()) {
        // Copy from CUDA to CPU using current stream and synchronize
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        cudaMemcpyAsync(output.data_ptr<float>(), x.data_ptr<float>(),
                        x.numel() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    } else if (x.device().is_cpu()) {
        std::memcpy(output.data_ptr<float>(), x.data_ptr<float>(), x.numel() * sizeof(float));
    } else {
        TORCH_CHECK(false, "Unsupported input device");
    }

    return output;
}

// Computes element-wise absolute value on CPU tensor.
torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cpu(), "Input must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    auto output = torch::empty_like(x);
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; ++i) {
        output_ptr[i] = std::abs(input_ptr[i]);
    }
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("to_device_forward", &to_device_forward, "Move tensor to CPU (copies if CUDA)", py::arg("x"));
    m.def("abs_forward", &abs_forward, "Element-wise absolute value", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
