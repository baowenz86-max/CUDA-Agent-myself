#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

#include <vector>

// Declare launchers from custom_ops.cu
extern "C" void softmax_forward_launcher(const float* input,
                                           float* output,
                                           int N, int C, int H, int W,
                                           cudaStream_t stream);

extern "C" void sort_forward_launcher(const float* input,
                                        float* out_values,
                                        int64_t* out_indices,
                                        int N, int C, int H, int W,
                                        cudaStream_t stream);

extern "C" void split_forward_launcher(const float* input,
                                         float** outputs,
                                         int N, int C, int H, int W,
                                         int split_size, int num_chunks,
                                         cudaStream_t stream);

static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

torch::Tensor softmax_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    // Ensure contiguous
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    check_cuda_error("softmax_forward_launcher failed");
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> sort_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto out_values = torch::empty_like(x);
    auto out_indices = torch::empty_like(x, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sort_forward_launcher(
        x.data_ptr<float>(),
        out_values.data_ptr<float>(),
        out_indices.data_ptr<int64_t>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    check_cuda_error("sort_forward_launcher failed");
    return std::make_tuple(out_values, out_indices);
}

std::vector<torch::Tensor> split_forward(torch::Tensor x, int64_t split_size, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(split_size > 0, "split_size must be positive");

    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t num_chunks = (C + split_size - 1) / split_size;
    std::vector<torch::Tensor> outputs;
    outputs.reserve(num_chunks);

    // Allocate output tensors with appropriate shapes
    for (int64_t i = 0; i < num_chunks; ++i) {
        int64_t chunk_len = std::min(split_size, C - i * split_size);
        outputs.push_back(torch::empty({N, chunk_len, H, W}, x.options()));
    }

    // Prepare pointer array
    std::vector<float*> output_ptrs;
    output_ptrs.reserve(num_chunks);
    for (auto& t : outputs) {
        output_ptrs.push_back(t.data_ptr<float>());
    }

    // Copy pointer array to device
    float** d_output_ptrs;
    cudaMalloc(&d_output_ptrs, num_chunks * sizeof(float*));
    TORCH_CHECK(d_output_ptrs != nullptr, "cudaMalloc failed");
    cudaMemcpy(d_output_ptrs, output_ptrs.data(), num_chunks * sizeof(float*), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    split_forward_launcher(
        x.data_ptr<float>(),
        d_output_ptrs,
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(split_size),
        static_cast<int>(num_chunks),
        stream
    );

    check_cuda_error("split_forward_launcher failed");

    cudaFree(d_output_ptrs);

    // Return as tuple (convert vector to tuple for pybind)
    // pybind11 will convert std::vector to tuple automatically? Actually we want tuple.
    // We can return std::vector and pybind will convert to list, but model_new.py expects tuple unpacking.
    // We'll return std::vector but model_new.py uses `x = cuda_extension.split_forward(...)` and then returns x, but split in Python returns tuple.
    // Since model_new.py just returns the result, it may be okay if it's a list? No, forward expects tuple of tensors.
    // We need to return a tuple. We can construct py::tuple manually, but easier: return std::vector<torch::Tensor> and rely on pybind's automatic conversion to tuple? Actually pybind converts std::vector to list, not tuple.
    // To match torch.split which returns tuple, we should return py::tuple.
    // However, we are in binding.cpp and can construct py::tuple from vector.
    py::tuple tuple_output(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        tuple_output[i] = outputs[i];
    }
    return outputs; // This will be converted to list; need to return py::tuple explicitly.
}

void register_custom_ops(pybind11::module& m) {
    m.def("softmax_forward", &softmax_forward, "Softmax forward along dim=1",
          py::arg("x"), py::arg("dim"));
    m.def("sort_forward", &sort_forward, "Stable sort forward along dim=1",
          py::arg("x"), py::arg("dim"));
    m.def("split_forward", &split_forward, "Split forward along dim=1",
          py::arg("x"), py::arg("split_size"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
