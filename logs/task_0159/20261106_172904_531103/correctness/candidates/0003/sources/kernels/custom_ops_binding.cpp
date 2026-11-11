#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void view_as_real_launcher(
    const void* input_data,
    float* output_data,
    const int64_t* dims,
    const int64_t* strides,
    cudaStream_t stream);

extern "C" void constant_pad_launcher(
    const float* input_data,
    float* output_data,
    int64_t B, int64_t H_in, int64_t W_in,
    int64_t stride_b, int64_t stride_h, int64_t stride_w,
    float pad_value,
    cudaStream_t stream);

// view_as_real forward
at::Tensor view_as_real_forward(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kComplexFloat, "Input dtype must be complex64");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, H, W)");

    auto B = x.size(0);
    auto H = x.size(1);
    auto W = x.size(2);

    auto output = at::empty({B, H, W, 2}, x.options().dtype(at::kFloat));

    // Prepare dims and strides arrays (strides in complex elements)
    std::vector<int64_t> dims = {B, H, W};
    std::vector<int64_t> strides = {x.stride(0), x.stride(1), x.stride(2)};

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    view_as_real_launcher(
        x.data_ptr(), // const void* (pointing to complex data, we interpret as float array)
        output.data_ptr<float>(),
        dims.data(),
        strides.data(),
        stream
    );

    return output;
}

// constant_pad forward
at::Tensor constant_pad_forward(at::Tensor x, py::object pad, double value = 0.0) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "Input dtype must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, H, W, C)");

    // Parse pad tuple/list. Expected: (left, right, top, bottom) for last two dims
    int64_t pad_left, pad_right, pad_top, pad_bottom;
    if (py::isinstance<py::tuple>(pad)) {
        auto t = pad.cast<py::tuple>();
        TORCH_CHECK(t.size() == 4, "pad tuple must have 4 elements");
        pad_left = t[0].cast<int64_t>();
        pad_right = t[1].cast<int64_t>();
        pad_top = t[2].cast<int64_t>();
        pad_bottom = t[3].cast<int64_t>();
    } else if (py::isinstance<py::list>(pad)) {
        auto l = pad.cast<py::list>();
        TORCH_CHECK(l.size() == 4, "pad list must have 4 elements");
        pad_left = py::cast<int64_t>(l[0]);
        pad_right = py::cast<int64_t>(l[1]);
        pad_top = py::cast<int64_t>(l[2]);
        pad_bottom = py::cast<int64_t>(l[3]);
    } else {
        // If int, replicate to all four sides
        int64_t p = pad.cast<int64_t>();
        pad_left = pad_right = pad_top = pad_bottom = p;
    }

    TORCH_CHECK(x.size(2) == 1024 && x.size(3) == 2, "Unsupported shape for this task");
    TORCH_CHECK(pad_left == 1 && pad_right == 1 && pad_top == 1 && pad_bottom == 1,
                "This binding only supports pad=(1,1,1,1)");

    auto B = x.size(0);
    auto H_in = x.size(1);
    auto W_in = x.size(2);
    auto C_in = x.size(3); // should be 2

    // Output shape: [B, H_in, W_in+2, C_in+2] = [B, H, W+2, 4]
    auto H_out = H_in;
    auto W_out = W_in + 2;
    auto C_out = C_in + 2;

    auto output = at::empty({B, H_out, W_out, C_out}, x.options());

    // Strides for input (first three dims; last dim stride assumed 1)
    auto stride_b = x.stride(0);
    auto stride_h = x.stride(1);
    auto stride_w = x.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    constant_pad_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H_in, W_in,
        stride_b, stride_h, stride_w,
        static_cast<float>(value),
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("view_as_real", &view_as_real_forward, "Convert complex tensor to real representation",
          py::arg("x"));
    m.def("constant_pad", &constant_pad_forward, "Constant pad last two dims by 1",
          py::arg("x"), py::arg("pad"), py::arg("value") = 0.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
