#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher，现在包含 shape 指针
extern "C" void sign_launcher(
    float* output,
    const float* input,
    int64_t numel,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* shape,
    int ndim,
    cudaStream_t stream
);

// sign_forward: 逐元素符号函数，保持 stride 不变
torch::Tensor sign_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input dtype must be float32");
    TORCH_CHECK(x.dim() > 0, "Input must have at least one dimension");

    // 获取输入的形状和 strides（保存在 host 上）
    const int ndim = x.dim();
    std::vector<int64_t> input_strides(ndim);
    std::vector<int64_t> output_strides(ndim);
    std::vector<int64_t> shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = x.size(i);
        input_strides[i] = x.stride(i);
        output_strides[i] = x.stride(i); // 输出与输入 stride 相同
    }

    // 分配输出张量，保持相同的 strides
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());
    
    // 将 shape 数组复制到 device 内存，以便 kernel 可以访问
    // 使用 torch::from_blob 创建一个临时的 CUDA tensor 来存放 shape
    auto shape_tensor = torch::empty({ndim}, torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
    // 将 shape 数据复制到 device
    // 注意：torch::empty 分配可能未初始化，我们需要使用 cudaMemcpy 或 tensor 的 copy_ 方法
    // 这里我们从 host 数据创建 CPU tensor 然后复制到 CUDA
    auto shape_cpu = torch::from_blob(shape.data(), {ndim}, torch::TensorOptions().dtype(torch::kInt64));
    shape_tensor.copy_(shape_cpu);

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher，传递 shape 的 device 指针
    sign_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        input_strides.data(),
        output_strides.data(),
        shape_tensor.data_ptr<int64_t>(),
        ndim,
        stream
    );

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("sign_forward", &sign_forward, "Sign function forward",
          py::arg("x"));
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
