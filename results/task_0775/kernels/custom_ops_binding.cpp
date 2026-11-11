#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of the CUDA launcher from custom_ops.cu
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
);

torch::Tensor batched_linear_forward(
    torch::Tensor x,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch_size, in_features]");

    TORCH_CHECK(!weights.empty(), "weights list must not be empty");
    TORCH_CHECK(!biases.empty(), "biases list must not be empty");
    TORCH_CHECK(weights.size() == biases.size(), "number of weights and biases must match");

    const int64_t batch_size = x.size(0);
    const int64_t in_features = x.size(1);
    const int64_t num_models = static_cast<int64_t>(weights.size());

    // Validate each weight and bias
    int64_t out_features = -1;
    for (int64_t i = 0; i < num_models; ++i) {
        const auto& w = weights[i];
        const auto& b = biases[i];
        TORCH_CHECK(w.is_cuda(), "weight ", i, " must be a CUDA tensor");
        TORCH_CHECK(w.dtype() == torch::kFloat32, "weight ", i, " must be float32");
        TORCH_CHECK(w.dim() == 2, "weight ", i, " must be 2D [out_features, in_features]");
        TORCH_CHECK(w.size(1) == in_features, "weight ", i, " in_features mismatch");
        if (out_features < 0) {
            out_features = w.size(0);
        } else {
            TORCH_CHECK(w.size(0) == out_features, "all weights must have same out_features");
        }

        TORCH_CHECK(b.is_cuda(), "bias ", i, " must be a CUDA tensor");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias ", i, " must be float32");
        TORCH_CHECK(b.dim() == 1, "bias ", i, " must be 1D [out_features]");
        TORCH_CHECK(b.size(0) == out_features, "bias ", i, " out_features mismatch");
    }

    // Stack weights and biases into contiguous tensors
    auto weights_stacked = torch::stack(weights, 0).contiguous();  // [num_models, out_features, in_features]
    auto biases_stacked = torch::stack(biases, 0).contiguous();    // [num_models, out_features]

    // Allocate output tensor: [num_models, batch_size, out_features]
    auto output = torch::empty({num_models, batch_size, out_features}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    batched_linear_forward_launcher(
        x.data_ptr<float>(),
        weights_stacked.data_ptr<float>(),
        biases_stacked.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(in_features),
        static_cast<int>(out_features),
        static_cast<int>(num_models),
        stream
    );

    return output;
}

void register_batched_linear_forward(pybind11::module& m) {
    m.def("batched_linear_forward", &batched_linear_forward,
          "Batched linear forward for multiple models",
          py::arg("x"), py::arg("weights"), py::arg("biases"));
}

REGISTER_BINDING(batched_linear_forward, register_batched_linear_forward);
