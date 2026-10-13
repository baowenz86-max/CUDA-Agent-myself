#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include "../binding_registry.h"

// 声明 .cu 导出的 launcher
extern "C" void upsample_nearest2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int scale,
    cudaStream_t stream);

extern "C" void one_hot_launcher(
    const int64_t* input,
    int64_t* output,
    int64_t total_spatial,
    int64_t num_classes,
    cudaStream_t stream);

// cuDNN 错误检查辅助宏
#define CUDNN_CHECK(expr)                                              \
  do {                                                                 \
    cudnnStatus_t status = (expr);                                     \
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS,                        \
                "cuDNN error: ", cudnnGetErrorString(status));         \
  } while (0)

// 辅助函数：获取当前 CUDA stream
static cudaStream_t getCurrentCUDAStream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

// 辅助函数：检查 tensor 是否满足要求
static void check_cuda_float_contiguous(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ConvTranspose2d 实现（使用 cuDNN）
torch::Tensor custom_conv_transpose2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {
  // 输入检查
  check_cuda_float_contiguous(input, "input");
  check_cuda_float_contiguous(weight, "weight");
  if (bias.defined()) {
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
  }
  TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
  TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C_in, C_out, kH, kW)");
  TORCH_CHECK(input.size(1) == weight.size(0), "input channels must match weight first dim");

  const int N = static_cast<int>(input.size(0));
  const int C_in = static_cast<int>(input.size(1));
  const int H_in = static_cast<int>(input.size(2));
  const int W_in = static_cast<int>(input.size(3));
  const int C_out = static_cast<int>(weight.size(1));
  const int kH = static_cast<int>(weight.size(2));
  const int kW = static_cast<int>(weight.size(3));

  // 输出尺寸计算
  int H_out = (H_in - 1) * static_cast<int>(stride) - 2 * static_cast<int>(padding) + kH + static_cast<int>(output_padding);
  int W_out = (W_in - 1) * static_cast<int>(stride) - 2 * static_cast<int>(padding) + kW + static_cast<int>(output_padding);
  TORCH_CHECK(H_out > 0 && W_out > 0, "invalid output size");

  // 分配输出 tensor
  auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

  // 获取 cuDNN handle
  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));
  CUDNN_CHECK(cudnnSetStream(handle, getCurrentCUDAStream()));

  // 创建 tensor descriptors
  cudnnTensorDescriptor_t x_desc, y_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H_in, W_in));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out));

  // 创建 filter descriptor
  cudnnFilterDescriptor_t w_desc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_in, C_out, kH, kW));

  // 创建卷积描述符
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc,
      static_cast<int>(padding), static_cast<int>(padding),
      static_cast<int>(stride), static_cast<int>(stride),
      1, 1,  // dilation
      CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT));

  // 选择前向算法（使用反向数据，因为是转置卷积）
  cudnnConvolutionFwdAlgo_t algo;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
      handle, y_desc, w_desc, conv_desc, x_desc,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

  // 计算工作空间大小
  size_t workspace_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      handle, y_desc, w_desc, conv_desc, x_desc, algo, &workspace_size));

  // 分配工作空间
  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaMalloc(&workspace, workspace_size);
  }

  // 执行卷积（注意：对于转置卷积，前向算法使用反向数据路径，
  // 即输入为 x（通常是较小空间），输出为 y（较大空间），
  // 这在 cuDNN 中通过使用 CUDNN_CONVOLUTION_BWD_DATA 或
  // 使用 CUDNN_CONVOLUTION_FWD 配合交换 input/output 和转置权重来实现。
  // 为简化，这里采用标准方式：将 ConvTranspose2d 视为对输入进行
  // 转置卷积，即等价于对梯度进行卷积。但 cuDNN 没有直接支持转置卷积，
  // 通常通过 cudnnConvolutionBackwardData 来实现。
  // 因此，下面使用 cudnnConvolutionBackwardData。
  // 注意：cuDNN 的 backward data 计算的是 dx，其中 dx shape 与输入 x 相同。
  // 对于 ConvTranspose2d，我们以 weight 和输出 grad（这里为 input）作为输入，
  // 计算 dx（即转置卷积输出）。
  // 所以需要将 input 作为 dy，输出作为 dx。
  // 使用 cudnnConvolutionBackwardData。
  // 但为了正确性，我们直接使用 cudnnConvolutionBackwardData 实现。
  // 注意：PyTorch 的 ConvTranspose2d 在 cuDNN 中通常使用 cudnnConvolutionBackwardData。
  CUDNN_CHECK(cudnnConvolutionBackwardData(
      handle,
      &one, // alpha = 1
      y_desc, // dyDesc (input)
      input.data_ptr<float>(),
      w_desc, // wDesc (weight)
      weight.data_ptr<float>(),
      conv_desc,
      algo,
      workspace, workspace_size,
      &zero, // beta = 0
      x_desc, // dxDesc (output)
      output.data_ptr<float>()));

  // 添加 bias
  if (bias.defined()) {
    // 使用 cudnnAddTensor 或手动 kernel? 这里使用 cudnnAddTensor
    cudnnTensorDescriptor_t bias_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
    float alpha = 1.0f;
    float beta = 1.0f;
    CUDNN_CHECK(cudnnAddTensor(handle, &alpha, bias_desc, bias.data_ptr<float>(), &beta, y_desc, output.data_ptr<float>()));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
  }

  // 释放资源
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
  if (workspace) {
    cudaFree(workspace);
  }
  CUDNN_CHECK(cudnnDestroy(handle));

  // 检查 CUDA 错误
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

  return output;
}

// Upsample Nearest2d
torch::Tensor custom_upsample_nearest2d(torch::Tensor input, double scale_factor) {
  check_cuda_float_contiguous(input, "input");
  TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

  const int N = static_cast<int>(input.size(0));
  const int C = static_cast<int>(input.size(1));
  const int H = static_cast<int>(input.size(2));
  const int W = static_cast<int>(input.size(3));
  const int scale = static_cast<int>(scale_factor);
  TORCH_CHECK(scale > 0, "scale_factor must be positive");

  // 输出尺寸
  const int out_H = H * scale;
  const int out_W = W * scale;
  auto output = torch::empty({N, C, out_H, out_W}, input.options());

  cudaStream_t stream = getCurrentCUDAStream();
  upsample_nearest2d_launcher(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      N, C, H, W, scale, stream);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
  return output;
}

// One-hot
torch::Tensor custom_one_hot(torch::Tensor target, int64_t num_classes) {
  TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
  TORCH_CHECK(target.dtype() == torch::kInt64, "target must be int64");
  TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
  TORCH_CHECK(target.dim() >= 1, "target must have at least 1 dim");
  TORCH_CHECK(num_classes > 0, "num_classes must be positive");

  // 输出形状: target.shape + [num_classes]
  std::vector<int64_t> out_shape(target.sizes().begin(), target.sizes().end());
  out_shape.push_back(num_classes);
  auto output = torch::empty(out_shape, target.options());

  int64_t total_spatial = target.numel();
  cudaStream_t stream = getCurrentCUDAStream();
  one_hot_launcher(
      target.data_ptr<int64_t>(),
      output.data_ptr<int64_t>(),
      total_spatial,
      num_classes,
      stream);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
  return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
  m.def("custom_conv_transpose2d", &custom_conv_transpose2d,
        "Custom ConvTranspose2d using cuDNN",
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
  m.def("custom_upsample_nearest2d", &custom_upsample_nearest2d,
        "Custom nearest neighbor upsample",
        py::arg("input"), py::arg("scale_factor"));
  m.def("custom_one_hot", &custom_one_hot,
        "Custom one_hot",
        py::arg("target"), py::arg("num_classes"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
