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

// Helper to check tensor is CUDA float32
static torch::Tensor check_cuda_float32(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    return t;
}

// custom_linear wrapper with optional bias compatible with operator_contract
// bias is nullable: if None, pass nullptr and skip bias addition
// We accept pybind11::object for bias to allow None and document as Optional
static torch::Tensor custom_linear_impl(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj) {
    
    input = check_cuda_float32(input.contiguous(), "input");
    weight = check_cuda_float32(weight.contiguous(), "weight");
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    TORCH_CHECK(in_features == weight.size(1), "input/weight dimension mismatch");
    
    torch::Tensor bias;
    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias = check_cuda_float32(bias.contiguous(), "bias");
        TORCH_CHECK(bias.numel() == out_features, "bias size mismatch");
        bias_ptr = bias.data_ptr<float>();
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

static torch::Tensor custom_kldiv_loss_impl(torch::Tensor x, torch::Tensor target) {
    x = check_cuda_float32(x.contiguous(), "x");
    target = check_cuda_float32(target.contiguous(), "target");
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

// Helper to convert a list/tensor of scalar parameters to contiguous 1D tensor
static torch::Tensor params_to_tensor(const pybind11::iterable& params_iter) {
    std::vector<torch::Tensor> tensors;
    for (auto item : params_iter) {
        torch::Tensor t = item.cast<torch::Tensor>();
        tensors.push_back(t);
    }
    TORCH_CHECK(!tensors.empty(), "params must not be empty");
    for (auto& t : tensors) {
        t = check_cuda_float32(t, "param");
        TORCH_CHECK(t.numel() == 1, "each param must have exactly one element");
    }
    return torch::stack(tensors).view(-1);
}

static torch::Tensor custom_l1_loss_sum_impl(torch::Tensor x, pybind11::object params_obj) {
    x = check_cuda_float32(x.contiguous(), "x");
    int batch_size = x.size(0);
    int num_cols = x.size(1);
    
    torch::Tensor params;
    if (pybind11::isinstance<torch::Tensor>(params_obj)) {
        params = params_obj.cast<torch::Tensor>();
        params = check_cuda_float32(params.contiguous(), "params");
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
    m.def("custom_linear", &custom_linear_impl,
          "Custom linear layer using cuBLAS",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());
    m.def("custom_kldiv_loss", &custom_kldiv_loss_impl,
          "Custom KLDiv loss with batchmean",
          py::arg("x"), py::arg("target"));
    m.def("custom_l1_loss_sum", &custom_l1_loss_sum_impl,
          "Custom sum of L1 losses",
          py::arg("x"), py::arg("params"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);