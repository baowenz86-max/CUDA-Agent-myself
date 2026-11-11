#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <tuple>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_embedding_launcher(const int64_t* indices,
                                          const float* weight,
                                          float* output,
                                          int64_t N,
                                          int64_t embedding_dim,
                                          cudaStream_t stream);

extern "C" void custom_split_launcher(const float* input,
                                      int64_t num_splits,
                                      int64_t dim,
                                      int64_t N,
                                      int64_t embedding_dim,
                                      const float** out_ptrs,
                                      int64_t* out_sizes,
                                      cudaStream_t stream);

// Wrapper for custom_embedding
torch::Tensor custom_embedding_forward(torch::Tensor indices, torch::Tensor weight) {
    // Validate inputs
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1-dimensional");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-dimensional");

    int64_t N = indices.size(0);
    int64_t embedding_dim = weight.size(1);
    if (N == 0) {
        // Return empty tensor with correct shape
        return torch::empty({0, embedding_dim}, weight.options());
    }

    // Allocate output
    auto output = torch::empty({N, embedding_dim}, weight.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_embedding_launcher(
        indices.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        embedding_dim,
        stream
    );

    return output;
}

// Wrapper for custom_split
std::tuple<torch::Tensor, torch::Tensor> custom_split_forward(torch::Tensor input,
                                                               int64_t num_splits,
                                                               int64_t dim) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "dim out of range");
    TORCH_CHECK(num_splits > 0, "num_splits must be positive");
    TORCH_CHECK(input.size(dim) % num_splits == 0, "split dimension must be divisible by num_splits");

    // For this task, we only support dim=0 and num_splits=2 per launcher contract
    TORCH_CHECK(dim == 0, "Only dim=0 is supported");
    TORCH_CHECK(num_splits == 2, "Only num_splits=2 is supported");

    int64_t N = input.size(0);
    int64_t embedding_dim = input.numel() / N; // total / rows

    // Prepare output pointers and sizes
    const float* out_ptrs[2];
    int64_t out_sizes[2];

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher to compute pointers/sizes
    custom_split_launcher(
        input.data_ptr<float>(),
        num_splits,
        dim,
        N,
        embedding_dim,
        out_ptrs,
        out_sizes,
        stream
    );

    // Check for null pointers (launcher should always set them for valid input)
    TORCH_CHECK(out_ptrs[0] != nullptr && out_ptrs[1] != nullptr, "split launcher returned null pointer");

    // Create tensor views sharing storage with input
    // Use torch::from_blob with the original storage of input to ensure proper sharing
    // Since input is contiguous, we can create views using as_strided or from_blob with deleter.
    auto input_untyped = input; // keep a reference to keep storage alive
    auto storage = input.storage(); // get the Storage, keeps data alive
    int64_t split_size = out_sizes[0];
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());

    torch::Tensor first = torch::from_blob(
        const_cast<float*>(out_ptrs[0]),
        {split_size, embedding_dim},
        [storage](void*) {},  // no-op deleter, storage keeps data alive
        options
    );

    torch::Tensor second = torch::from_blob(
        const_cast<float*>(out_ptrs[1]),
        {split_size, embedding_dim},
        [storage](void*) {},
        options
    );

    return std::make_tuple(first, second);
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_embedding", &custom_embedding_forward,
          "Custom embedding lookup",
          py::arg("indices"),
          py::arg("weight"));
    m.def("custom_split", &custom_split_forward,
          "Split input along dim=0 into two equal halves as views",
          py::arg("input"),
          py::arg("num_splits"),
          py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
