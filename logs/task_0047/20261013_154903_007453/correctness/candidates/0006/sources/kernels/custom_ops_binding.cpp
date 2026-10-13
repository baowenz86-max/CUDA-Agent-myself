#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_linear_launcher(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    cudaStream_t stream);

extern "C" void custom_kldiv_loss_launcher(
    const float* x, const float* target, float* output,
    int batch_size, int num_cols, cudaStream_t stream);

extern "C" void custom_l1_loss_sum_launcher(
    const float* x, const float* params, float* output,
    int batch_size, int num_cols, cudaStream_t stream);

// Helper to check tensor is CUDA float32 and make contiguous
torch::Tensor check_and_contiguous(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return t;
}

// custom_linear wrapper
torch::Tensor custom_linear(torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias_opt) {
    input = check_and_contiguous(input, "input");
    weight = check_and_contiguous(weight, "weight");

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        bias = check_and_contiguous(bias, "bias");
        bias_ptr = bias.data_ptr<float>();
    }

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    TORCH_CHECK(in_features == weight.size(1), "input/weight dimension mismatch");
    if (bias_ptr != nullptr) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.numel() == out_features, "bias size mismatch");
    }

    auto output = torch::empty({batch_size, out_features}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_linear_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        stream
    );
    return output;
}

// custom_kldiv_loss wrapper
torch::Tensor custom_kldiv_loss(torch::Tensor x, torch::Tensor target) {
    x = check_and_contiguous(x, "x");
    target = check_and_contiguous(target, "target");
    TORCH_CHECK(x.sizes() == target.sizes(), "x and target must have same shape");

    int batch_size = x.size(0);
    int num_cols = x.size(1);
    auto output = torch::empty({}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_kldiv_loss_launcher(
        x.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_cols,
        stream
    );
    return output;
}

// Helper to convert a ParameterList (or list of tensors) to a contiguous tensor
torch::Tensor params_to_tensor(const pybind11::iterable& params_iter) {
    std::vector<torch::Tensor> tensors;
    for (auto item : params_iter) {
        torch::Tensor t = item.cast<torch::Tensor>();
        tensors.push_back(t);
    }
    TORCH_CHECK(!tensors.empty(), "params must not be empty");
    for (auto& t : tensors) {
        TORCH_CHECK(t.is_cuda(), "params must be CUDA tensors");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "params must be float32");
        TORCH_CHECK(t.numel() == 1, "each param must have exactly one element");
    }
    return torch::stack(tensors).view(-1);
}

// custom_l1_loss_sum wrapper: accepts either a tensor or a list of tensors
torch::Tensor custom_l1_loss_sum(torch::Tensor x, pybind11::object params_obj) {
    x = check_and_contiguous(x, "x");
    int batch_size = x.size(0);
    int num_cols = x.size(1);

    torch::Tensor params;
    if (pybind11::isinstance<torch::Tensor>(params_obj)) {
        params = params_obj.cast<torch::Tensor>();
        params = check_and_contiguous(params, "params");
    } else if (pybind11::isinstance<pybind11::iterable>(params_obj)) {
        params = params_to_tensor(params_obj.cast<pybind11::iterable>());
    } else {
        TORCH_CHECK(false, "params must be a Tensor or iterable of Tensors");
    }
    TORCH_CHECK(params.size(0) == num_cols, "params length must equal number of columns");

    auto output = torch::empty({}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_l1_loss_sum_launcher(
        x.data_ptr<float>(),
        params.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_cols,
        stream
    );
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_linear", &custom_linear,
          "Custom linear layer using cuBLAS",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());
    m.def("custom_kldiv_loss", &custom_kldiv_loss,
          "Custom KLDiv loss with batchmean",
          py::arg("x"), py::arg("target"));
    m.def("custom_l1_loss_sum", &custom_l1_loss_sum,
          "Custom sum of L1 losses",
          py::arg("x"), py::arg("params"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
