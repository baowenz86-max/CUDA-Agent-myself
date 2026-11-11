#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

#include <vector>
#include <cstdint>

// Launcher declarations from custom_ops.cu
extern "C" void isin_custom_launcher(const float* x, const float* elements, bool* output,
                                    int64_t numel, int64_t elements_size, cudaStream_t stream);

extern "C" void split_with_sizes_custom_launcher(
    const float* x,
    int64_t num_outputs,
    float** outputs,
    int64_t* output_numels,
    int64_t* input_offsets,
    cudaStream_t stream);

// isin_custom: output[i] = (x[i] in elements)
torch::Tensor isin_custom(torch::Tensor x, torch::Tensor elements) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(elements.is_cuda(), "elements must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(elements.dtype() == torch::kFloat32, "elements must be float32");
    TORCH_CHECK(elements.dim() == 1, "elements must be 1-dimensional");
    TORCH_CHECK(elements.is_contiguous(), "elements must be contiguous");

    auto output = torch::empty(x.sizes(), x.options().dtype(torch::kBool));
    if (x.numel() == 0) {
        return output;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    isin_custom_launcher(
        x.data_ptr<float>(),
        elements.data_ptr<float>(),
        output.data_ptr<bool>(),
        x.numel(),
        elements.numel(),
        stream
    );
    // Error checking after launch is optional; launcher already returns void.
    return output;
}

// split_with_sizes_custom: split x along dim=1 according to split_sizes
torch::Tensor split_with_sizes_custom(torch::Tensor x, int64_t split_sizes) {
    // Note: split_sizes is typed as int64_t, but the Python call passes a list.
    // We need to handle the list by accepting py::list, but contract says int64_t.
    // The following is an intentional mismatch; operator contract expects pybind to accept list.
    TORCH_CHECK(false, "split_with_sizes_custom should be called with a list; see registration");
    return torch::Tensor();
}

// Actual wrapper for split_with_sizes_custom that accepts a Python list of ints.
// Since operator_contract listed parameter as int (singular), but actual Python call uses a list,
// we must use a std::vector<int64_t> or py::list at binding. We'll define the real function here.
// To avoid duplicate registration, we won't use the above stub; we'll define a different name
torch::Tensor split_with_sizes_custom_impl(torch::Tensor x, std::vector<int64_t> split_sizes) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
    int64_t dim = 1; // fixed by reference
    TORCH_CHECK(dim < x.dim(), "dim out of range");

    int64_t sum_sizes = 0;
    for (auto s : split_sizes) {
        sum_sizes += s;
    }
    TORCH_CHECK(sum_sizes == x.size(dim), "split_sizes must sum to x.size(dim)");

    int64_t num_outputs = split_sizes.size();
    std::vector<torch::Tensor> outputs;
    outputs.reserve(num_outputs);

    // Allocate outputs as contiguous tensors with same options as x
    auto options = x.options();
    for (int64_t i = 0; i < num_outputs; ++i) {
        std::vector<int64_t> sizes = x.sizes().vec();
        sizes[dim] = split_sizes[i];
        outputs.push_back(torch::empty(sizes, options));
    }

    // Prepare launcher arrays
    std::vector<float*> output_ptrs(num_outputs);
    std::vector<int64_t> output_numels(num_outputs);
    std::vector<int64_t> input_offsets(num_outputs);

    int64_t offset = 0;
    for (int64_t i = 0; i < num_outputs; ++i) {
        output_ptrs[i] = outputs[i].data_ptr<float>();
        output_numels[i] = outputs[i].numel();
        input_offsets[i] = offset;
        offset += outputs[i].numel();
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    split_with_sizes_custom_launcher(
        x.data_ptr<float>(),
        num_outputs,
        output_ptrs.data(),
        output_numels.data(),
        input_offsets.data(),
        stream
    );

    // Convert vector to tuple for return
    py::tuple tuple_outputs(num_outputs);
    for (int64_t i = 0; i < num_outputs; ++i) {
        tuple_outputs[i] = outputs[i];
    }
    return torch::Tensor(tuple_outputs);
}

// Registration function for pybind
void register_custom_ops(pybind11::module& m) {
    m.def("isin_custom", &isin_custom, "Check if elements of x are in elements",
          py::arg("x"), py::arg("elements"));
    m.def("split_with_sizes_custom", &split_with_sizes_custom_impl,
          "Split x along dim=1 according to split_sizes",
          py::arg("x"), py::arg("split_sizes"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
