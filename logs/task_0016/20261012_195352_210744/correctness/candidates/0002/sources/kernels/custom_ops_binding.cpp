#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void addmm_forward_launcher(const float* bias,
                                       const float* A,
                                       const float* B,
                                       float* C,
                                       int M, int N, int K,
                                       cudaStream_t stream);

extern "C" void unsqueeze_forward_launcher(const float* input,
                                           float* output,
                                           int total_elements,
                                           int dim,
                                           cudaStream_t stream);

extern "C" void interpolate_linear_forward_launcher(const float* input,
                                                    float* output,
                                                    int input_height,
                                                    int input_width,
                                                    int output_width,
                                                    float scale_factor,
                                                    cudaStream_t stream);

// ---------------- addmm_forward ----------------
torch::Tensor addmm_forward(torch::Tensor bias, torch::Tensor mat1, torch::Tensor mat2) {
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(mat1.is_cuda(), "mat1 must be a CUDA tensor");
    TORCH_CHECK(mat2.is_cuda(), "mat2 must be a CUDA tensor");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
    TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(mat1.dtype() == torch::kFloat32, "mat1 must be float32");
    TORCH_CHECK(mat2.dtype() == torch::kFloat32, "mat2 must be float32");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2D");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be 2D");
    int N = bias.size(0);
    int M = mat1.size(0);
    int K = mat1.size(1);
    TORCH_CHECK(mat2.size(0) == N && mat2.size(1) == K,
                "mat2 must have shape [N, K] = [hidden_size, input_size]");
    auto C = torch::empty({M, N}, mat1.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    addmm_forward_launcher(bias.data_ptr<float>(),
                           mat1.data_ptr<float>(),
                           mat2.data_ptr<float>(),
                           C.data_ptr<float>(),
                           M, N, K,
                           stream);
    return C;
}

// ---------------- unsqueeze_forward ----------------
torch::Tensor unsqueeze_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(dim == 0, "only dim=0 is supported");
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    unsqueeze_forward_launcher(input.data_ptr<float>(),
                               output.data_ptr<float>(),
                               total_elements,
                               static_cast<int>(dim),
                               stream);
    // Unsqueeze to shape [1, *input.sizes()]
    std::vector<int64_t> out_shape;
    out_shape.push_back(1);
    for (auto s : input.sizes()) out_shape.push_back(s);
    return output.view(out_shape);
}

// ---------------- interpolate_linear_forward ----------------
torch::Tensor interpolate_linear_forward(torch::Tensor input, double scale_factor) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [1, M, N]");
    TORCH_CHECK(input.size(0) == 1, "batch dimension must be 1 for this task");
    int64_t M = input.size(1);
    int64_t N = input.size(2);
    int64_t out_N = static_cast<int64_t>(N * scale_factor);
    TORCH_CHECK(out_N > 0, "output width must be positive");
    auto output = torch::empty({1, M, out_N}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    interpolate_linear_forward_launcher(input.data_ptr<float>(),
                                        output.data_ptr<float>(),
                                        static_cast<int>(M),
                                        static_cast<int>(N),
                                        static_cast<int>(out_N),
                                        static_cast<float>(scale_factor),
                                        stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("addmm_forward", &addmm_forward, "AddMM forward",
          py::arg("bias"), py::arg("mat1"), py::arg("mat2"));
    m.def("unsqueeze_forward", &unsqueeze_forward, "Unsqueeze forward",
          py::arg("input"), py::arg("dim"));
    m.def("interpolate_linear_forward", &interpolate_linear_forward, "Linear interpolation forward",
          py::arg("input"), py::arg("scale_factor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
