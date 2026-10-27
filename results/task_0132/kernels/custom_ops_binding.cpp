#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void constant_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int left, int right, int top, int bottom,
    float value,
    cudaStream_t stream);

extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int left, int right, int top, int bottom,
    cudaStream_t stream);

// Helper to convert Python padding tuple or int to left, right, top, bottom
static void parse_padding(const pybind11::object& padding, int& left, int& right, int& top, int& bottom) {
    if (pybind11::isinstance<pybind11::int_>(padding)) {
        int p = padding.cast<int>();
        left = right = top = bottom = p;
    } else if (pybind11::isinstance<pybind11::tuple>(padding)) {
        auto t = padding.cast<pybind11::tuple>();
        TORCH_CHECK(t.size() == 4, "padding must be an int or a tuple of 4 ints");
        left = t[0].cast<int>();
        right = t[1].cast<int>();
        top = t[2].cast<int>();
        bottom = t[3].cast<int>();
    } else {
        TORCH_CHECK(false, "padding must be an int or a tuple of 4 ints");
    }
    TORCH_CHECK(left >= 0 && right >= 0 && top >= 0 && bottom >= 0, "padding values must be non-negative");
}

// constant_pad2d: accepts CPU or CUDA input, outputs CUDA tensor
// If input is CPU, it is first copied to a temporary CUDA tensor, then kernel executes.
torch::Tensor constant_pad2d(torch::Tensor x, pybind11::object padding) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "constant_pad2d input must be float32");
    TORCH_CHECK(x.is_contiguous(), "constant_pad2d input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "constant_pad2d input must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "input dimensions must be positive");
    TORCH_CHECK(N <= INT_MAX && C <= INT_MAX && H <= INT_MAX && W <= INT_MAX, "dimensions too large");

    int left, right, top, bottom;
    parse_padding(padding, left, right, top, bottom);
    int64_t outH = H + top + bottom;
    int64_t outW = W + left + right;
    TORCH_CHECK(outH <= INT_MAX && outW <= INT_MAX, "output dimensions too large");

    // Ensure input is on CUDA (copy from CPU if necessary)
    torch::Tensor x_cuda;
    if (x.is_cuda()) {
        x_cuda = x;
    } else {
        x_cuda = x.to(torch::kCUDA);
    }

    // Allocate output on CUDA
    auto output = torch::empty({N, C, outH, outW}, x_cuda.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    constant_pad2d_launcher(
        x_cuda.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C),
        static_cast<int>(H), static_cast<int>(W),
        left, right, top, bottom,
        0.0f,  // constant pad value is 0
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "constant_pad2d kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// reflection_pad2d: input is CUDA float32, output is CUDA float32
torch::Tensor reflection_pad2d(torch::Tensor x, pybind11::object padding) {
    TORCH_CHECK(x.is_cuda(), "reflection_pad2d input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "reflection_pad2d input must be float32");
    TORCH_CHECK(x.is_contiguous(), "reflection_pad2d input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "reflection_pad2d input must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "input dimensions must be positive");
    TORCH_CHECK(N <= INT_MAX && C <= INT_MAX && H <= INT_MAX && W <= INT_MAX, "dimensions too large");
    TORCH_CHECK(H >= 1 && W >= 1, "reflection padding requires H,W >= 1");

    int left, right, top, bottom;
    parse_padding(padding, left, right, top, bottom);
    int64_t outH = H + top + bottom;
    int64_t outW = W + left + right;
    TORCH_CHECK(outH <= INT_MAX && outW <= INT_MAX, "output dimensions too large");

    // Allocate CUDA output tensor
    auto output = torch::empty({N, C, outH, outW}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    reflection_pad2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C),
        static_cast<int>(H), static_cast<int>(W),
        left, right, top, bottom,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "reflection_pad2d kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("constant_pad2d", &constant_pad2d,
          "ConstantPad2d with CPU or CUDA input and CUDA output",
          pybind11::arg("x"), pybind11::arg("padding"));
    m.def("reflection_pad2d", &reflection_pad2d,
          "ReflectionPad2d with CUDA input and CUDA output",
          pybind11::arg("x"), pybind11::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
