#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void cast_float_launcher(
    const uint8_t* input,
    float* output,
    int total,
    cudaStream_t stream);

torch::Tensor cast_float_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cast_float_launcher(
        input.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

void register_cast_float_forward(pybind11::module& m) {
    m.def("cast_float_forward", &cast_float_forward, "Cast uint8 to float32");
}

REGISTER_BINDING(cast_float, register_cast_float_forward);
