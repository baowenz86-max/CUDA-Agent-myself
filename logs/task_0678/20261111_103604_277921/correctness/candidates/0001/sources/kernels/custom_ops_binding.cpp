#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void flip_cat_launcher(
    float* out,
    const float* x1,
    const float* x2,
    int64_t total_out,
    const int64_t* shape,
    int ndim,
    int64_t dim,
    cudaStream_t stream
);

extern "C" void flip_launcher(
    float* out,
    const float* in,
    int64_t total,
    const int64_t* shape,
    int ndim,
    int64_t dim,
    cudaStream_t stream
);

extern "C" void cat_launcher(
    float* out,
    const float* x1,
    const float* x2,
    int64_t total_out,
    const int64_t* shape,
    int ndim,
    int64_t dim,
    cudaStream_t stream
);

namespace {

// Check common tensor requirements for all ops
void check_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// Validate dim for 4D tensors used in this task
void check_dim(int64_t dim) {
    TORCH_CHECK(dim >= 0 && dim < 4, "dim must be in [0, 3] for 4D tensors, got ", dim);
}

// Copy shape to std::vector<int64_t> for launcher
std::vector<int64_t> get_shape_vec(const torch::Tensor& t) {
    auto sizes = t.sizes();
    return std::vector<int64_t>(sizes.begin(), sizes.end());
}

} // namespace

// flip_forward: flip tensor along dim
torch::Tensor flip_forward(int64_t dim, torch::Tensor x) {
    check_tensor(x, "x");
    check_dim(dim);
    
    auto shape = get_shape_vec(x);
    int64_t total = x.numel();
    
    auto out = torch::empty_like(x);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    flip_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        total,
        shape.data(),
        static_cast<int>(shape.size()),
        dim,
        stream
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "flip_launcher failed");
    return out;
}

// cat_forward: concatenate two tensors along dim
torch::Tensor cat_forward(int64_t dim, torch::Tensor x1, torch::Tensor x2) {
    check_tensor(x1, "x1");
    check_tensor(x2, "x2");
    check_dim(dim);
    
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");
    
    auto shape = get_shape_vec(x1);
    int64_t total_out = x1.numel() * 2;
    
    // output shape: double the dim dimension
    std::vector<int64_t> out_shape(shape);
    out_shape[dim] *= 2;
    
    auto out = torch::empty(out_shape, x1.options());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cat_launcher(
        out.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        total_out,
        shape.data(),
        static_cast<int>(shape.size()),
        dim,
        stream
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "cat_launcher failed");
    return out;
}

// flip_cat_forward: flip both tensors and concatenate
torch::Tensor flip_cat_forward(int64_t dim, torch::Tensor x1, torch::Tensor x2) {
    check_tensor(x1, "x1");
    check_tensor(x2, "x2");
    check_dim(dim);
    
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");
    
    auto shape = get_shape_vec(x1);
    int64_t total_out = x1.numel() * 2;
    
    // output shape: double the dim dimension
    std::vector<int64_t> out_shape(shape);
    out_shape[dim] *= 2;
    
    auto out = torch::empty(out_shape, x1.options());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    flip_cat_launcher(
        out.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        total_out,
        shape.data(),
        static_cast<int>(shape.size()),
        dim,
        stream
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "flip_cat_launcher failed");
    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("flip_forward", &flip_forward, "Flip tensor along dimension",
          py::arg("dim"), py::arg("x"));
    m.def("cat_forward", &cat_forward, "Concatenate two tensors along dimension",
          py::arg("dim"), py::arg("x1"), py::arg("x2"));
    m.def("flip_cat_forward", &flip_cat_forward, "Flip two tensors and concatenate",
          py::arg("dim"), py::arg("x1"), py::arg("x2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
