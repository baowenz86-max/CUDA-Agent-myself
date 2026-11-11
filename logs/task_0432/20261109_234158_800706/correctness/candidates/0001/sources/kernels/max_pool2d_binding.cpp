#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// cuDNN error checking helper
#define CHECK_CUDNN(expr)                                                       \
  do {                                                                         \
    cudnnStatus_t status = (expr);                                             \
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS,                                \
                "cuDNN error: ", cudnnGetErrorString(status));                 \
  } while (0)

torch::Tensor max_pool2d_forward(torch::Tensor x, int64_t kernel_size) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // MaxPool2d with stride=kernel_size, padding=0, dilation=1
    int64_t out_h = (H - kernel_size) / kernel_size + 1;
    int64_t out_w = (W - kernel_size) / kernel_size + 1;
    TORCH_CHECK(out_h > 0 && out_w > 0, "Invalid output dimensions for pooling");

    auto output = torch::empty({N, C, out_h, out_w}, x.options());

    // Handle non-contiguous input by making contiguous if needed
    auto x_contig = x.contiguous();

    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnPoolingDescriptor_t pooling_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           N, C, H, W));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           N, C, out_h, out_w));

    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pooling_desc,
                                            CUDNN_POOLING_MAX,
                                            CUDNN_PROPAGATE_NAN,
                                            kernel_size, kernel_size,
                                            0, 0,
                                            kernel_size, kernel_size));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    CHECK_CUDNN(cudnnSetStream(handle, stream));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnPoolingForward(handle,
                                    pooling_desc,
                                    &alpha,
                                    x_desc,
                                    x_contig.data_ptr<float>(),
                                    &beta,
                                    y_desc,
                                    output.data_ptr<float>()));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pooling_desc));
    CHECK_CUDNN(cudnnDestroy(handle));

    return output;
}

void register_max_pool2d_binding(pybind11::module& m) {
    m.def("max_pool2d_forward", &max_pool2d_forward,
          "MaxPool2d forward using cuDNN",
          py::arg("x"), py::arg("kernel_size"));
}

REGISTER_BINDING(max_pool2d, register_max_pool2d_binding);
