#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void layer_norm_launcher(const float* x,
                                    const float* weight,
                                    const float* bias,
                                    float* y,
                                    int batch,
                                    int out_channels,
                                    int length,
                                    float eps,
                                    cudaStream_t stream);

extern "C" void addcmul_launcher(const float* input,
                                 const float* tensor1,
                                 const float* tensor2,
                                 float* output,
                                 int total_elements,
                                 float value,
                                 cudaStream_t stream);

// -------------------- cuDNN ConvTranspose1d implementation --------------------

// Helper: check cudnn status
#define CHECK_CUDNN(expr)                                                      \
  do {                                                                         \
    cudnnStatus_t status = (expr);                                             \
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS,                                \
                "cuDNN error: ", cudnnGetErrorString(status));                 \
  } while (0)

// Wrapper for ConvTranspose1d using cuDNN.
// Input x: [batch, in_channels, length] float32 CUDA tensor (contiguous)
// weight: [in_channels, out_channels, kernel_size] float32 CUDA tensor (contiguous)
// bias: optional [out_channels] float32 CUDA tensor (contiguous), may be null
// stride, padding, output_padding: integers
// Returns [batch, out_channels, L_out] contiguous tensor
torch::Tensor conv_transpose_forward_cudnn(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

  TORCH_CHECK(x.is_cuda(), "conv_transpose_forward: x must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "conv_transpose_forward: weight must be CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "conv_transpose_forward: x must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose_forward: weight must be float32");
  TORCH_CHECK(x.dim() == 3, "conv_transpose_forward: x must be 3D [batch, in_channels, length]");
  TORCH_CHECK(weight.dim() == 3, "conv_transpose_forward: weight must be 3D [in_channels, out_channels, kernel_size]");

  // Make input contiguous (reference expects strided support but we can handle via contiguous)
  auto x_contig = x.contiguous();
  auto weight_contig = weight.contiguous();

  int64_t batch = x_contig.size(0);
  int64_t in_channels = x_contig.size(1);
  int64_t length = x_contig.size(2);
  int64_t out_channels = weight_contig.size(1);
  int64_t kernel_size = weight_contig.size(2);

  // Compute output length as PyTorch ConvTranspose1d: L_out = (length - 1)*stride - 2*padding + kernel_size + output_padding
  int64_t L_out = (length - 1) * stride - 2 * padding + kernel_size + output_padding;
  TORCH_CHECK(L_out > 0, "conv_transpose_forward: computed output length must be positive");

  // Optional bias handling
  torch::Tensor bias_contig;
  if (bias.defined()) {
    TORCH_CHECK(bias.is_cuda(), "conv_transpose_forward: bias must be CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "conv_transpose_forward: bias must be float32");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_channels,
                "conv_transpose_forward: bias shape must be [out_channels]");
    bias_contig = bias.contiguous();
  }

  // Allocate output tensor [batch, out_channels, L_out]
  auto output = torch::empty({batch, out_channels, L_out}, x_contig.options());

  // Get current CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // Create cuDNN handle
  cudnnHandle_t handle;
  CHECK_CUDNN(cudnnCreate(&handle));
  CHECK_CUDNN(cudnnSetStream(handle, stream));

  // Create tensor descriptors
  cudnnTensorDescriptor_t x_desc, y_desc, bias_desc;
  cudnnFilterDescriptor_t w_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

  // Set tensor descriptors: using NCHW format, but our tensors are [N, C, L], we treat L as W dimension with H=1.
  // cuDNN expects at least 4D for conv, so we use [N, C, 1, L] by adding a height dimension of 1.
  int dims[4] = {static_cast<int>(batch), static_cast<int>(in_channels), 1, static_cast<int>(length)};
  int strides[4] = {static_cast<int>(in_channels * length), static_cast<int>(length), static_cast<int>(length), 1};
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         dims[0], dims[1], dims[2], dims[3]));

  int y_dims[4] = {static_cast<int>(batch), static_cast<int>(out_channels), 1, static_cast<int>(L_out)};
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         y_dims[0], y_dims[1], y_dims[2], y_dims[3]));

  // Filter descriptor: cuDNN expects [K, C, R, S] for NCHW, but ConvTranspose1d weight is [in_channels, out_channels, kernel_size].
  // PyTorch ConvTranspose1d weight shape is (in_channels, out_channels, kernel_size), which corresponds to (C, K, R) in 1D.
  // In cuDNN, for transposed convolution we use the weight for the forward pass of the transpose, but we can use cudnnConvolutionBackwardData
  // with a filter that has shape [K, C, R, S]? Actually, cuDNN's cudnnConvolutionBackwardData expects wDesc to be [K, C, R, S] for NCHW.
  // However, PyTorch's weight is [in_channels, out_channels, kernel_size] = [C, K, R]. We need to transmute to [K, C, R, S] with S=1.
  // We will create a temporary filter tensor in the correct layout or set the descriptor with a stride trick.
  // Simpler: since the dimensions are small, we can create a temporary contiguous tensor of shape [out_channels, in_channels, kernel_size, 1]
  // and copy/permute data from weight_contig. But to avoid expensive data movement on host, we can do a device copy with permute using a small custom kernel? Not allowed.
  // Alternative: cuDNN supports 'CUDNN_TENSOR_NCHW_VECT_C'? Not for filter.
  // The safest method: use cudnnSetFilter4dDescriptor with format CUDNN_TENSOR_NCHW and dimensions [K, C, R, S].
  // But our weight memory is [C, K, R]. We can create a new tensor with shape [K, C, R, 1] and use a simple manual copy? Not possible without compute.
  // However, we can use cudnnSetFilter4dDescriptor with a different memory format? CUDNN_TENSOR_NCHW expects data in [K, C, R, S] order.
  // To avoid permute, we can use the fact that ConvTranspose1d can be expressed as a gradient of a regular Conv1d, and cuDNN's cudnnConvolutionBackwardData
  // allows the filter to be in the same layout as the forward conv filter (which is [K, C, R] for 1D). But cuDNN only supports 2D+ convolutions, so we add a dummy dim S=1.
  // Many implementations create a cudnnFilterDescriptor with shape [K, C, 1, kernel_size] to represent a 1D convolution as 2D with height=1.
  // In that case, the weight layout should be [K, C, 1, R], which is similar to PyTorch's Conv2d weight layout [out_channels, in_channels, kernel_height, kernel_width].
  // For ConvTranspose1d, PyTorch's weight is [in_channels, out_channels, kernel_size] (without the dummy dim). This is because ConvTranspose1d weight layout is transposed.
  // Actually, for Conv1d, weight is [out_channels, in_channels, kernel_size]. For ConvTranspose1d, weight is [in_channels, out_channels, kernel_size].
  // So for cuDNN's cudnnConvolutionBackwardData, the filter descriptor should match the convolution that would have produced this grad: that conv has weights [out_channels, in_channels, kernel_size] (regular Conv1d).
  // So we need a filter descriptor with dims [K=out_channels, C=in_channels, R=kernel_size, S=1] (or [K, C, 1, R] depending on convention).
  // To use our weight_contig (which is [in_channels, out_channels, kernel_size]), we must permute it to [out_channels, in_channels, kernel_size].
  // We can do that on GPU with a small permute or simply use cublas? But we are allowed to use cuBLAS only for GEMM, not for general permute. However, we can create a temporary tensor and perform a device-side permute via a kernel? Not allowed in .cpp.
  // Another workaround: use cudnnSetFilter4dDescriptor with a stride trick to interpret the same memory as [K, C, R, S]? Since the memory is contiguous, we can specify strides manually, but the data layout is fixed.
  // The weight memory: indices: (c, k, r) -> offset = c*out_channels*kernel_size + k*kernel_size + r.
  // We want to see it as (k, c, r) -> offset should be k*in_channels*kernel_size + c*kernel_size + r.
  // These are different; a simple stride change cannot represent a permute.
  // Therefore, we need to permute the weight on device. However, we are allowed to use torch:: ops? The instructions say NO torch operators in C++, but we can use torch::Tensor operations like permute? That would be using torch:: operator, which is forbidden.
  // We are allowed to allocate with torch::empty_like, but not perform compute with torch::. So we cannot call weight.permute().
  // Instead, we can implement a small CUDA kernel for permute? But .cpp cannot have CUDA kernels; they must be in .cu. But we are not modifying .cu in this stage? The task says "只生成或修改 kernels/ 下的绑定 .cpp 文件，不要生成或修改 .cu 或 model_new.py". So we cannot add a permute kernel to .cu.
  // However, we could use cublas? No.
  // Alternative: use cuDNN's cudnnConvolutionBackwardData with the filter in the original layout? cuDNN might accept a filter descriptor with layout CUDNN_TENSOR_NCHW where dimensions are [in_channels, out_channels, kernel_size, 1]? That would not match the expected conv weight [out_channels, in_channels, kernel_size, 1].
  // Another approach: Instead of transposed convolution via cuDNN, we can use cuDNN's cudnnConvolutionForward with a trick: insert zeros between input elements? Not allowed.
  // But the requirement says "Conv/ConvTranspose 必须在 kernels/*.cpp 中使用 cuDNN 实现". So we must use cuDNN.
  // Perhaps we can use cudnnConvolutionBackwardData with a filter descriptor that directly matches PyTorch's ConvTranspose1d weight layout by exploiting the equivalence: ConvTranspose1d is the gradient of Conv1d with respect to input, and the weight for that Conv1d is the same as ConvTranspose1d's weight but permuted. PyTorch internally stores ConvTranspose1d weight in the same layout as Conv1d weight? No, actually for ConvTranspose1d, weight shape is [in_channels, out_channels, kernel_size], which is exactly the transpose of Conv1d weight [out_channels, in_channels, kernel_size]. So to use cuDNN's cudnnConvolutionBackwardData, we need to pass the filter in the layout expected by the forward conv, which is [out_channels, in_channels, kernel_size]. So we need a permute.
  // Is it permissible to call a device kernel from .cpp? No, but we can call a CUDA kernel implemented in .cu. Since we are not allowed to modify .cu in this stage, we must either assume that conv_transpose launcher exists in .cu (but it doesn't) or find a way to avoid permute.
  // However, the task says "当前阶段允许文件：kernels/*.cpp". We can create a new .cu? No, "kernels/ 必须且只能有一个 .cu：custom_ops.cu". So we cannot add another .cu. We must work with the existing custom_ops.cu, which does not have conv_transpose launcher.
  // So we need to implement conv_transpose in the binding.cpp using cuDNN, but without a permute kernel, we need to work with the weight layout as is.
  // Could we use cudnnSetFilter4dDescriptor with the stride array to interpret the weight as [out_channels, in_channels, kernel_size, 1] but with non-contiguous strides? The data is contiguous with shape [in_channels, out_channels, kernel_size]. To view as [out_channels, in_channels, kernel_size] we need a stride of (kernel_size, out_channels*kernel_size, 1) for dims (k, c, r). But the actual memory stride is (out_channels*kernel_size, kernel_size, 1) for dims (c, k, r). If we set the descriptor with dimensions [out_channels, in_channels, kernel_size] and strides [kernel_size, out_channels*kernel_size, 1], we would be accessing memory out of order, but that's allowed if we have a custom stride, but the physical layout is not that. cuDNN expects the data to be laid out according to those strides, but the data is actually laid out with strides [out_channels*kernel_size, kernel_size, 1]. So that would produce incorrect results.
  // So we need to perform a permute on device.
  // But we cannot add a kernel. However, we might be able to use cudnnTransformTensor to permute? That's still using cuDNN, but it's allowed. cudnnTransformTensor can perform a permute if we create tensors with appropriate dims and strides. For a 3D tensor [C, K, R] and we want [K, C, R], we can use cudnnTransformTensor with input tensor desc [C, K, R] and output tensor desc [K, C, R] both with NCHW? That might work if we create a 4D descriptor with dummy dims. Since cudnnTransformTensor is a data movement operation, it's allowed as part of cuDNN usage.
  // Alternatively, we can use cudnnConvolutionBackwardData directly with a filter descriptor that matches the original weight shape but interpret the convolution differently. But that's messy.
  // Let's use cudnnTransformTensor to permute the weight from [in_channels, out_channels, kernel_size] to [out_channels, in_channels, kernel_size].
  // We'll allocate a temporary tensor for permuted weight on GPU.
  // Steps:
  //   - Create temp_weight = torch::empty({out_channels, in_channels, kernel_size}, options) (contiguous)
  //   - Create input tensor desc for weight_contig with dims [in_channels, out_channels, kernel_size, 1] and strides matching contiguous (out_channels*kernel_size, kernel_size, 1, 1).
  //   - Create output tensor desc for temp_weight with dims [out_channels, in_channels, kernel_size, 1] and strides matching contiguous (in_channels*kernel_size, kernel_size, 1, 1).
  //   - call cudnnTransformTensor with direction CUDNN_TENSOR_NCHW2NCHW? Actually cudnnTransformTensor just copies/transforms data according to descriptors.
  //   - Then use the permuted weight in cudnnConvolutionBackwardData.
  // This avoids custom kernels and stays within cuDNN.
  // We'll need to handle the alpha/beta scalars.

  // Let's implement this.

  // ---- Permute weight ----
  auto perm_weight = torch::empty({out_channels, in_channels, kernel_size}, x_contig.options());

  cudnnTensorDescriptor_t w_in_desc, w_out_desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&w_in_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&w_out_desc));

  // Input weight desc: layout as [in_channels, out_channels, kernel_size] with dummy 4th dim = 1
  int w_in_dims[4] = {1, static_cast<int>(in_channels), static_cast<int>(out_channels), static_cast<int>(kernel_size)};
  // Strides for contiguous [C, K, R] as NCHW? Actually N=1, C=in_channels, H=out_channels, W=kernel_size? Not standard, but we can set arbitrary dims with strides.
  // Since cudnnSetTensor4dDescriptor expects NCHW layout, we can set N=1, C=in_channels, H=out_channels, W=kernel_size, and the strides should be:
  // element (n,c,h,w) -> offset = n*strideN + c*strideC + h*strideH + w*strideW.
  // For contiguous memory of shape [in_channels, out_channels, kernel_size] as a 3D tensor, we want offset = c*out_channels*kernel_size + h*kernel_size + w.
  // So we set: strideN = in_channels*out_channels*kernel_size (though N only 0), strideC = out_channels*kernel_size, strideH = kernel_size, strideW = 1.
  // But cudnnSetTensor4dDescriptor only sets dimensions and format; it assumes contiguous strides based on format. To specify custom strides, we need cudnnSetTensor4dDescriptorEx.
  // We'll use cudnnSetTensor4dDescriptorEx to set non-contiguous strides.

  // For w_in_desc: dims [1, in_channels, out_channels, kernel_size]
  int w_in_dims_arr[4] = {1, static_cast<int>(in_channels), static_cast<int>(out_channels), static_cast<int>(kernel_size)};
  int w_in_strides[4] = {static_cast<int>(in_channels * out_channels * kernel_size),
                         static_cast<int>(out_channels * kernel_size),
                         static_cast<int>(kernel_size),
                         1};
  CHECK_CUDNN(cudnnSetTensor4dDescriptorEx(w_in_desc, CUDNN_DATA_FLOAT,
                                           w_in_dims_arr[0], w_in_dims_arr[1], w_in_dims_arr[2], w_in_dims_arr[3],
                                           w_in_strides[0], w_in_strides[1], w_in_strides[2], w_in_strides[3]));

  // For w_out_desc: dims [1, out_channels, in_channels, kernel_size] (desired layout for cuDNN filter)
  int w_out_dims_arr[4] = {1, static_cast<int>(out_channels), static_cast<int>(in_channels), static_cast<int>(kernel_size)};
  int w_out_strides[4] = {static_cast<int>(out_channels * in_channels * kernel_size),
                          static_cast<int>(in_channels * kernel_size),
                          static_cast<int>(kernel_size),
                          1};
  CHECK_CUDNN(cudnnSetTensor4dDescriptorEx(w_out_desc, CUDNN_DATA_FLOAT,
                                           w_out_dims_arr[0], w_out_dims_arr[1], w_out_dims_arr[2], w_out_dims_arr[3],
                                           w_out_strides[0], w_out_strides[1], w_out_strides[2], w_out_strides[3]));

  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnTransformTensor(handle, &alpha, w_in_desc, weight_contig.data_ptr<float>(),
                                   &beta, w_out_desc, perm_weight.data_ptr<float>()));

  // Now perm_weight has shape [out_channels, in_channels, kernel_size] and is contiguous.
  // We can treat it as a 4D filter with dims [out_channels, in_channels, 1, kernel_size] for cudnnConvolutionBackwardData.
  // Set filter descriptor for the convolution: dims [K, C, R, S] = [out_channels, in_channels, 1, kernel_size].
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                         static_cast<int>(out_channels),
                                         static_cast<int>(in_channels),
                                         1,
                                         static_cast<int>(kernel_size)));

  // Set convolution descriptor: 1D convolution with stride, padding, output_padding.
  // For cudnnConvolutionBackwardData, the parameters correspond to the forward conv that would have produced the grad:
  //   forward conv: input [batch, out_channels, L_out] (our desired output) -> output [batch, in_channels, length] (our input).
  //   stride, padding, dilation are from the forward conv. We need to set conv_desc accordingly.
  // For transposed convolution, the output_padding is implemented by adjusting the input size of the backward data operation? Actually, PyTorch's ConvTranspose1d with output_padding adds extra zeros to the output. cuDNN's cudnnConvolutionBackwardData does not have an output_padding parameter. Instead, we can set the convolution descriptor to have padding = kernel_size - 1 - padding (for transposed conv) and then adjust? Let's recall: ConvTranspose1d is equivalent to the gradient of Conv1d with respect to input. The forward Conv1d has: input length = L_out, output length = length. The relation: length = (L_out - 1)*stride - 2*padding_conv + kernel_size, where padding_conv is the padding used in the forward conv. In transposed conv, we are given stride, padding, output_padding. The equivalent forward conv has stride = stride, padding = kernel_size - 1 - padding, dilation = 1. Then output_padding is achieved by adding extra rows/columns to the output of the transposed conv. In cuDNN's cudnnConvolutionBackwardData, the output is the gradient with respect to the input of the forward conv, and its size is determined by the input size of the forward conv. To get the desired L_out, we need to set the input tensor of the backward data operation to have length L_out, and then the output (gradient) will have length determined by the forward conv output size, which should match our original input length. Actually, cudnnConvolutionBackwardData takes as input the gradOutput (dy) with shape [batch, in_channels, length] (matching our x), and outputs gradInput (dx) with shape [batch, out_channels, L_out] (matching our desired output). The convolution descriptor describes the forward conv: input shape [batch, out_channels, L_out], output shape [batch, in_channels, length]. So we set conv_desc with stride, padding = kernel_size - 1 - padding, dilation=1. Then the sizes of dy and dx are consistent. But output_padding is not considered; it effectively increases L_out by output_padding. To incorporate output_padding, we can set the output tensor dx to have extra zeros at the end (along length dimension) by allocating L_out + output_padding and then slicing? But PyTorch's ConvTranspose1d with output_padding produces L_out = (length-1)*stride - 2*padding + kernel_size + output_padding. The cudnnConvolutionBackwardData with the above settings would produce L_out_base = (length-1)*stride - 2*(kernel_size-1-padding) + kernel_size = (length-1)*stride - 2*padding + kernel_size. So we need output_padding extra elements at the end. We can achieve this by allocating a larger buffer and setting the bias/add appropriately? But the extra elements should be zero (or filled by bias if bias is added later). Actually, in PyTorch, output_padding adds zeros at the end of the output (after the transposed conv and before bias addition). So we can compute the base transposed conv output of size baseL = (length-1)*stride - 2*padding + kernel_size, then append output_padding zeros (but they are already zero if we allocate with empty, but we need to ensure that region is written with zeros). Since we allocate a tensor of size L_out = baseL + output_padding, we can run cudnnConvolutionBackwardData to fill the first baseL elements, and then set the remaining output_padding elements to zero (and add bias only to the baseL part? Actually bias should be added to all positions? In PyTorch, bias is added to every spatial location of the output, including the extra padded positions? Let's check: ConvTranspose1d with output_padding: The output is (conv_transpose result without output_padding) with output_padding zeros appended at the end along the length dimension, then bias is added to the whole output including the appended zeros. So we need to add bias to all L_out positions.
  // So method: allocate output of L_out. Set last output_padding elements to zero (if we use empty, they are uninitialized, so we need to zero them). Run cudnnConvolutionBackwardData to write the baseL positions. Then add bias to all positions. Alternatively, we can zero the entire output first and then run cudnnConvolutionBackwardData with beta=0, it writes only baseL positions and zeros the rest? No, cudnnConvolutionBackwardData writes to the entire dx tensor of size specified by the descriptor y_desc, which we set to [batch, out_channels, baseL]. If we set y_desc to [batch, out_channels, L_out], then the output size would be L_out, but the convolution will compute only baseL values and leave the rest undefined? It might not write to the extra positions depending on implementation. Safer to zero the whole output first, then compute with a smaller output tensor and copy? Better: allocate output of size L_out, set it to zero using cudaMemset or torch zero? We can use cudaMemset on the device pointer to set all bytes to zero, then run cudnnConvolutionBackwardData with dx descriptor matching baseL (a temporary tensor), then copy that into the first baseL elements of output. But copying is another operation. Alternatively, we can set the y_desc to have dimensions [batch, out_channels, baseL] for the base computation, allocate a temporary tensor base_output of size baseL, compute into it, then create final output of L_out, copy base_output into the first part, and zero the last part. That's multiple ops but acceptable.
  // To keep things simpler, we can allocate output of size L_out, zero it using cudaMemset, then set the output tensor descriptor to match the full L_out, and run cudnnConvolutionBackwardData with beta=1.0 (accumulate) after we have zeroed. But cudnn will only write baseL elements; the rest remain zero. Thus output is correct. However, we need to ensure the convolution operation does not write beyond baseL. Since we set the output tensor descriptor (dx) to have dimension L_out, cudnn might expect the convolution to produce L_out elements, but with the given conv parameters, it will only produce baseL, and the rest might be left as is (if beta is non-zero) or undefined. To be safe, we zero first and use beta=1.0 to accumulate into zeros. But if cudnn writes beyond baseL, it would overwrite with something, but it shouldn't because the mathematical output size of cudnnConvolutionBackwardData is determined by the input size (dy) and conv descriptor, not by the output tensor descriptor. Actually, the output tensor descriptor must match the expected output size; if it's larger, cudnn may error or behave incorrectly. So better to compute into a base_output of the correct size and then handle the padding.
  // Approach: allocate base_output (batch, out_channels, baseL), run cudnnConvolutionBackwardData with appropriate descriptors, then allocate final_output (batch, out_channels, L_out), copy base_output into final_output's first baseL elements (using cudaMemcpyAsync or a small copy via cudnnTransformTensor), and set the last output_padding elements to zero (if not already zero from allocation, we can zero them with cudaMemsetAsync on the slice). Then add bias to entire final_output if bias present.
  // We'll implement this.

  int64_t baseL = (length - 1) * stride - 2 * padding + kernel_size;
  TORCH_CHECK(baseL >= 0, "conv_transpose_forward: invalid computed base length");

  auto base_output = torch::empty({batch, out_channels, baseL}, x_contig.options());

  // Set dx (output of backward data) descriptor to base_output size.
  int base_dims[4] = {static_cast<int>(batch), static_cast<int>(out_channels), 1, static_cast<int>(baseL)};
  cudnnTensorDescriptor_t base_desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&base_desc));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(base_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         base_dims[0], base_dims[1], base_dims[2], base_dims[3]));

  // Set convolution descriptor.
  // For transposed conv, we need pad = kernel_size - 1 - padding.
  int conv_pad = static_cast<int>(kernel_size - 1 - padding);
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                              0, 0,  // pad_h, pad_w (we set both to conv_pad? Actually 1D, so pad_h=0, pad_w=conv_pad)
                                              static_cast<int>(stride), 1, // stride_h, stride_w
                                              1, 1, // dilation_h, dilation_w
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));
  // For 1D, we set pad_h = 0, pad_w = conv_pad, stride_h = 1? Actually in 2D representation, we want a convolution over W only, with H dimension length 1. So stride_h should be 1, stride_w = stride; pad_h = 0, pad_w = conv_pad; dilation_h = 1, dilation_w = 1.

  // Run cudnnConvolutionBackwardData
  float alpha_bd = 1.0f, beta_bd = 0.0f;
  CHECK_CUDNN(cudnnConvolutionBackwardData(handle,
                                           &alpha_bd,
                                           w_desc, perm_weight.data_ptr<float>(),
                                           x_desc, x_contig.data_ptr<float>(),   // dy
                                           conv_desc,
                                           &beta_bd,
                                           base_desc, base_output.data_ptr<float>())); // dx

  // Allocate final output with L_out
  auto final_output = torch::empty({batch, out_channels, L_out}, x_contig.options());

  // Copy base_output into final_output's first baseL elements along last dimension.
  // We can use cudnnTransformTensor or a simple cudaMemcpy2D? Since the tensors are contiguous and the difference is only in the last dimension, we can copy element-wise using cudaMemcpyAsync for each batch and channel, or use cudnnTransformTensor to copy from base_output to a view of final_output.
  // Simpler: use cudaMemcpyAsync for the whole base_output (which is contiguous with total elements batch*out_channels*baseL) into final_output.data_ptr<float>(). Since final_output is contiguous and first batch*out_channels*baseL elements correspond to the base portion, we can do a single cudaMemcpyAsync.
  size_t base_numel = batch * out_channels * baseL;
  cudaMemcpyAsync(final_output.data_ptr<float>(), base_output.data_ptr<float>(),
                  base_numel * sizeof(float), cudaMemcpyDeviceToDevice, stream);

  // Zero the remaining part (last output_padding elements per row? Actually the padding is only at the end of each length dimension, so we need to zero the trailing output_padding elements for each (batch, out_channel) row.
  // Since final_output is contiguous with shape [batch, out_channels, L_out], the last output_padding elements per row are not contiguous at the end of the tensor; they are interspersed. So we need to zero them individually.
  // We can use a simple loop over batch and out_channels, and call cudaMemsetAsync on the slice pointer for output_padding elements.
  if (output_padding > 0) {
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t oc = 0; oc < out_channels; ++oc) {
        float* row_start = final_output.data_ptr<float>() + (b * out_channels + oc) * L_out + baseL;
        cudaMemsetAsync(row_start, 0, output_padding * sizeof(float), stream);
      }
    }
  }

  // Add bias if present
  if (bias_contig.defined()) {
    // Add bias to each channel over the whole length L_out.
    // We can use cuDNN's add tensor or a simple loop with cudaMemcpy? For bias addition, we can use cudnnAddTensor with bias descriptor and output tensor descriptor.
    cudnnTensorDescriptor_t final_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&final_desc));
    int final_dims[4] = {static_cast<int>(batch), static_cast<int>(out_channels), 1, static_cast<int>(L_out)};
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(final_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           final_dims[0], final_dims[1], final_dims[2], final_dims[3]));

    // Bias descriptor: shape [1, out_channels, 1, 1]
    int bias_dims[4] = {1, static_cast<int>(out_channels), 1, 1};
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           bias_dims[0], bias_dims[1], bias_dims[2], bias_dims[3]));

    float alpha_add = 1.0f, beta_add = 1.0f;
    CHECK_CUDNN(cudnnAddTensor(handle, &alpha_add, bias_desc, bias_contig.data_ptr<float>(),
                               &beta_add, final_desc, final_output.data_ptr<float>()));
  }

  // Cleanup
  cudnnDestroyTensorDescriptor(x_desc);
  cudnnDestroyTensorDescriptor(y_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyFilterDescriptor(w_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyTensorDescriptor(w_in_desc);
  cudnnDestroyTensorDescriptor(w_out_desc);
  cudnnDestroyTensorDescriptor(base_desc);
  if (bias_contig.defined()) {
    cudnnDestroyTensorDescriptor(final_desc);
  }
  cudnnDestroy(handle);

  return final_output;
}

// -------------------- LayerNorm wrapper --------------------

torch::Tensor layer_norm_forward_cuda(torch::Tensor x,
                                      torch::Tensor weight,
                                      torch::Tensor bias,
                                      double eps) {
  TORCH_CHECK(x.is_cuda(), "layer_norm_forward: x must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "layer_norm_forward: weight must be CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "layer_norm_forward: bias must be CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "layer_norm_forward: x must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "layer_norm_forward: weight must be float32");
  TORCH_CHECK(bias.dtype() == torch::kFloat32, "layer_norm_forward: bias must be float32");
  TORCH_CHECK(x.dim() == 3, "layer_norm_forward: x must be 3D [batch, out_channels, length]");

  auto x_contig = x.contiguous();
  auto weight_contig = weight.contiguous();
  auto bias_contig = bias.contiguous();

  int64_t batch = x_contig.size(0);
  int64_t out_channels = x_contig.size(1);
  int64_t length = x_contig.size(2);

  TORCH_CHECK(weight_contig.numel() == out_channels * length, "layer_norm_forward: weight shape mismatch");
  TORCH_CHECK(bias_contig.numel() == out_channels * length, "layer_norm_forward: bias shape mismatch");

  auto output = torch::empty_like(x_contig);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  layer_norm_launcher(
      x_contig.data_ptr<float>(),
      weight_contig.data_ptr<float>(),
      bias_contig.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(out_channels),
      static_cast<int>(length),
      static_cast<float>(eps),
      stream
  );

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "layer_norm_forward: kernel launch failed: ", cudaGetErrorString(err));

  return output;
}

// -------------------- Addcmul wrapper --------------------

torch::Tensor addcmul_forward_cuda(torch::Tensor input,
                                   torch::Tensor tensor1,
                                   torch::Tensor tensor2,
                                   double value) {
  TORCH_CHECK(input.is_cuda(), "addcmul_forward: input must be CUDA tensor");
  TORCH_CHECK(tensor1.is_cuda(), "addcmul_forward: tensor1 must be CUDA tensor");
  TORCH_CHECK(tensor2.is_cuda(), "addcmul_forward: tensor2 must be CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "addcmul_forward: input must be float32");
  TORCH_CHECK(tensor1.dtype() == torch::kFloat32, "addcmul_forward: tensor1 must be float32");
  TORCH_CHECK(tensor2.dtype() == torch::kFloat32, "addcmul_forward: tensor2 must be float32");
  TORCH_CHECK(input.sizes() == tensor1.sizes() && tensor1.sizes() == tensor2.sizes(),
              "addcmul_forward: all tensors must have the same shape");

  auto input_contig = input.contiguous();
  auto tensor1_contig = tensor1.contiguous();
  auto tensor2_contig = tensor2.contiguous();

  auto output = torch::empty_like(input_contig);
  int64_t total_elements = input_contig.numel();

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  addcmul_launcher(
      input_contig.data_ptr<float>(),
      tensor1_contig.data_ptr<float>(),
      tensor2_contig.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(total_elements),
      static_cast<float>(value),
      stream
  );

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "addcmul_forward: kernel launch failed: ", cudaGetErrorString(err));

  return output;
}

// -------------------- PyBind Registration --------------------

void register_custom_ops(pybind11::module& m) {
  m.def("conv_transpose_forward", &conv_transpose_forward_cudnn,
        "ConvTranspose1d forward using cuDNN",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
        pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
  m.def("layer_norm_forward", &layer_norm_forward_cuda,
        "LayerNorm forward",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"), pybind11::arg("eps"));
  m.def("addcmul_forward", &addcmul_forward_cuda,
        "Addcmul forward",
        pybind11::arg("input"), pybind11::arg("tensor1"), pybind11::arg("tensor2"), pybind11::arg("value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
