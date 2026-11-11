#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int batch,
    int channels,
    int length,
    long long in_stride_batch,
    long long in_stride_channel,
    long long in_stride_length,
    long long out_stride_batch,
    long long out_stride_channel,
    long long out_stride_length,
    cudaStream_t stream
);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Wrapper for add_bias: output = input + bias (broadcast over channels)
torch::Tensor add_bias(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "add_bias: input must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "add_bias: bias must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "add_bias: input must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "add_bias: bias must be float32");
    TORCH_CHECK(input.dim() == 3, "add_bias: input must be 3D (batch, channels, length)");
    TORCH_CHECK(bias.dim() == 2, "add_bias: bias must be 2D (channels, 1)");
    TORCH_CHECK(bias.size(0) == input.size(1), "add_bias: bias channels must match input channels");
    TORCH_CHECK(bias.size(1) == 1, "add_bias: bias second dimension must be 1");
    TORCH_CHECK(bias.is_contiguous(), "add_bias: bias must be contiguous");

    auto output = torch::empty_like(input);
    if (input.numel() == 0) {
        return output;
    }

    // Get strides in elements
    auto strides = input.strides();
    long long in_stride_batch = strides[0];
    long long in_stride_channel = strides[1];
    long long in_stride_length = strides[2];

    auto out_strides = output.strides();
    long long out_stride_batch = out_strides[0];
    long long out_stride_channel = out_strides[1];
    long long out_stride_length = out_strides[2];

    int batch = input.size(0);
    int channels = input.size(1);
    int length = input.size(2);

    cudaStream_t stream = getCurrentStream();

    add_bias_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        length,
        in_stride_batch,
        in_stride_channel,
        in_stride_length,
        out_stride_batch,
        out_stride_channel,
        out_stride_length,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "add_bias: CUDA kernel launch failed");
    return output;
}

// Wrapper for conv_transpose1d_forward using cuDNN
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(input.is_cuda(), "conv_transpose1d_forward: input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose1d_forward: weight must be CUDA tensor");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt->is_cuda(), "conv_transpose1d_forward: bias must be CUDA tensor");
    }
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "conv_transpose1d_forward: input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "conv_transpose1d_forward: weight must be float32");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt->scalar_type() == torch::kFloat32, "conv_transpose1d_forward: bias must be float32");
    }
    TORCH_CHECK(input.dim() == 3, "conv_transpose1d_forward: input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "conv_transpose1d_forward: weight must be 3D (C_in, C_out, kernel_size)");
    TORCH_CHECK(weight.is_contiguous(), "conv_transpose1d_forward: weight must be contiguous");

    int64_t batch = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_length = input.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    int64_t out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_length > 0, "conv_transpose1d_forward: output length must be positive");

    TORCH_CHECK(weight.size(0) == in_channels, "conv_transpose1d_forward: weight in_channels dimension mismatch");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt->numel() == out_channels, "conv_transpose1d_forward: bias size must match out_channels");
    }

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnCreate failed");

    // Set stream
    cudaStream_t stream = getCurrentStream();
    status = cudnnSetStream(cudnn_handle, stream);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnSetStream failed");

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    status = cudnnCreateTensorDescriptor(&input_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnCreateTensorDescriptor input failed");
    status = cudnnCreateTensorDescriptor(&output_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnCreateTensorDescriptor output failed");
    status = cudnnCreateTensorDescriptor(&bias_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnCreateTensorDescriptor bias failed");
    status = cudnnCreateFilterDescriptor(&filter_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnCreateFilterDescriptor failed");
    status = cudnnCreateConvolutionDescriptor(&conv_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnCreateConvolutionDescriptor failed");

    // Set tensor descriptors: cuDNN uses NCHW format, for 1D we treat length as width and height=1
    // Input: [batch, in_channels, 1, in_length]
    status = cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch,
        in_channels,
        1,
        in_length
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnSetTensor4dDescriptor input failed");

    // Output: [batch, out_channels, 1, out_length]
    status = cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch,
        out_channels,
        1,
        out_length
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnSetTensor4dDescriptor output failed");

    // Bias descriptor: [1, out_channels, 1, 1]
    status = cudnnSetTensor4dDescriptor(
        bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,
        out_channels,
        1,
        1
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnSetTensor4dDescriptor bias failed");

    // Filter descriptor: weight shape [in_channels, out_channels, kernel_size]
    // cuDNN expects filter as [out_channels, in_channels/groups, kernel_height, kernel_width]
    // For ConvTranspose1d, weight layout is [in_channels, out_channels, kernel_size]
    // We need to set filter descriptor accordingly. cuDNN transposed conv uses weight shape [in_channels, out_channels, kernel_height, kernel_width]
    // So kernel_height = 1, kernel_width = kernel_size
    status = cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        in_channels,  // k = number of input feature maps
        out_channels, // c = number of output feature maps
        1,            // kernel height
        kernel_size   // kernel width
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnSetFilter4dDescriptor failed");

    // Convolution descriptor
    // For ConvTranspose, we set mode as CUDNN_CROSS_CORRELATION, and specify padding, stride, and output_padding via adjust
    status = cudnnSetConvolution2dDescriptor(
        conv_desc,
        0,             // pad_h = 0 (padding is in width dimension for 1D)
        padding,       // pad_w = padding
        stride,        // stride_h = stride (not used as height=1, but set for consistency)
        stride,        // stride_w = stride
        1,             // dilation_h
        1,             // dilation_w
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnSetConvolution2dDescriptor failed");

    // Adjust convolution descriptor for output padding
    int pad_h = 0;
    int pad_w = output_padding;
    status = cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h,
        padding,
        stride,
        stride,
        1,
        1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: adjust output_padding failed (via descriptor)");
    // Actually output_padding is handled by cudnnGetConvolution2dForwardOutputDim and selection of algorithm, but we can specify it via cudnnSetConvolution2dDescriptor? 
    // cuDNN's convolution descriptor does not have output_padding field. Instead, transposed conv output size is determined by formula: (input - 1)*stride - 2*padding + kernel + output_padding.
    // In cuDNN, output size for transposed convolution is computed by cudnnGetConvolution2dForwardOutputDim using conv_desc and input/output descriptors, but output_padding can be added by adjusting the convolution descriptor's pad? There is a function cudnnSetConvolution2dDescriptor with padA and padB? Actually, cuDNN v7 has cudnnSetConvolution2dDescriptor with padA, padB? No, it's cudnnSetConvolution2dDescriptor(handle, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, mode, computeType). output_padding is not directly available. However, we can incorporate output_padding by increasing the padding after convolution? Typically, for transposed conv in PyTorch, output_padding adds extra zeros to the output, but cuDNN's cudnnConvolutionBackwardData (which implements transposed conv) uses a different convolution descriptor where output_padding can be set via cudnnSetConvolution2dDescriptor with pad = padding + output_padding? No, that's not correct.
    // The correct approach: For transposed convolution, output size is derived from the convolution descriptor. To achieve the desired output length, we can set the convolution descriptor's padding to (padding - output_padding) and then adjust? But this is getting complicated.
    // Simpler: We can use cudnnConvolutionBackwardData, which computes transposed convolution. The output size is determined by the convolution descriptor and input/output tensor descriptors. The formula used by cuDNN for backward data output size is: output = (input-1)*stride - 2*pad + kernel. If we need output_padding additional, we can set pad = padding - output_padding? Actually PyTorch's ConvTranspose2d output size is (in-1)*stride - 2*pad + kernel + output_padding. cuDNN's backward data uses the same formula without output_padding. To get the extra output_padding, we can adjust the padding to be padding - output_padding? No, because output_padding is typically 0 or 1 for stride > 1. If we set pad = padding - output_padding, then output size increases by 2*output_padding? That would be wrong. Actually output_padding only adds at the end (or start), not symmetrically. 
    // Given the complexity, a common approach is to set the output tensor descriptor to the desired output size directly, and set the convolution descriptor padding to padding, and rely on cuDNN's algorithm to handle the rest? But cuDNN's cudnnGetConvolution2dForwardOutputDim would compute output size without output_padding and might reject if output tensor descriptor is larger. So we need to set the output tensor descriptor to a size that cuDNN expects, and then handle output_padding separately by cropping or padding. 
    // However, in our case, output_padding=1, stride=2, padding=1, kernel=3. Output length computed by formula: (10240-1)*2 - 2*1 + 3 + 1 = 20480. Without output_padding it would be 20479. So we need the extra 1. 
    // One way: We can pad the input on the right by output_padding before calling cuDNN? No.
    // Another way: Use cudnnConvolutionBackwardData with a convolution descriptor that has padding = padding, stride = stride, and then after the operation, crop or pad the output to desired length. But the output from cuDNN will be of length 20479. We can then extend it by one element on the right with zero (or something) and then add bias? But the correct transposed conv result with output_padding should have additional zeros at the end, then bias, then external bias. Actually PyTorch's ConvTranspose1d with output_padding just adds extra zeros to the output before adding bias? Let's check: nn.ConvTranspose1d with output_padding adds zeros to the output at the end, but the convolution operation itself uses the weight to compute values up to the output size without output_padding. The extra output_padding values are filled with bias only (if bias present) or zeros (if no bias). But actually, the convolution output is computed for the full output size including output_padding, but the last output_padding positions are zero because the filter doesn't cover them. So we can simply compute the conv_transpose without output_padding (output size 20479) using cuDNN, then create a tensor of size 20480, copy the result into the first 20479 positions, and set the last position to zero. Then add bias (conv bias) to all positions? Wait, PyTorch's implementation: ConvTranspose1d forward = convolution_transpose(input, weight, bias, stride, padding, output_padding, groups, dilation). The output size is (in-1)*stride - 2*padding + kernel + output_padding. The operation is equivalent to: for each output position, sum over input positions and kernel positions where (output_index + padding - kernel_index) % stride == 0 and within input range. For the extra output_padding positions at the end, there are no valid input contributions, so they are zero before bias addition. Then bias is added to all positions. So we can implement this by computing the core convolution (without output_padding) using cuDNN, then padding the output with zeros, and then adding the bias. 
    // To simplify, we'll use cuDNN to compute convolution with output size as if output_padding=0, then create output tensor of desired size and copy the computed values into the beginning, leaving the last output_padding positions zero. Then add bias (if provided) to all positions. This should match PyTorch's behavior for ConvTranspose1d.
    // But we also need to handle the case where output_padding might be > stride? Usually output_padding < stride, so only a few positions at the end are zero. In our case, output_padding=1.
    // So plan:
    // 1. Compute conv_transpose without output_padding using cudnnConvolutionBackwardData, output size = (in-1)*stride - 2*padding + kernel.
    // 2. Allocate final output tensor of size (batch, out_channels, out_length).
    // 3. Copy the computed result into final output[:, :, :core_length] using cudaMemcpyAsync or a simple kernel (or torch::cat? but we can't use torch ops). We'll use cudaMemcpy2DAsync or just a custom small kernel to copy? We can use cudaMemcpyAsync for each batch and channel? That would be many copys. Instead, we can allocate a bigger tensor and use cuDNN with a larger output descriptor that includes output_padding? Some cuDNN versions allow the output tensor size to be larger than the formula, and the extra positions are ignored? I'm not sure. Safer approach: compute core output, then allocate final output and copy the core region using a simple copy kernel (or use cudaMemcpy2DAsync for each batch/channel). Since out_channels=16, batch=128, we have 2048 copys of a line of length 20479. That's many but maybe acceptable? Better to use a custom kernel to copy and zero-pad in one pass. But we are told not to implement CUDA kernels in .cpp. However, we can call an existing launcher from .cu if we declare it. But custom_ops.cu does not have a copy kernel. We could add a simple copy kernel to custom_ops.cu, but we are only allowed to modify .cpp in this stage? The instruction says: "当前阶段允许文件：kernels/*.cpp" and we should not generate or modify .cu. So we cannot add a copy kernel to .cu. Therefore, we need a way to do the padding without custom kernel. Could we use cudaMemsetAsync to zero the whole final output, then use cudaMemcpy2DAsync to copy the core lines? For each batch and channel, we can copy a 1D line of length core_length from the core_output to final_output with appropriate offsets. This can be done with a loop over batch*channels, using cudaMemcpyAsync (or cudaMemcpy2DAsync). That's 2048 async memcpys, which might be okay but not ideal. Alternatively, we can allocate final_output and core_output as a contiguous tensor and then use cudaMemcpyAsync for the entire data excluding the last element of each line, but that requires a custom copy pattern, not a simple linear copy. cudaMemcpy2DAsync can copy 2D arrays with pitch, but our data is 3D: [batch, channels, length]. We can treat it as a 2D array of size (batch*channels) x length. Then core_output has width core_length, final_output has width out_length = core_length + output_padding. We can copy each row (length core_length) to the corresponding row in final_output using cudaMemcpy2DAsync with appropriate pitch and width. That is a single call! Because cudaMemcpy2DAsync copies a 2D array of height = batch*channels, width = core_length, from source with pitch = core_length*sizeof(float) (if contiguous), to destination with pitch = out_length*sizeof(float). That should work if both tensors are contiguous and we treat the leading dimension as height. So we can do:
    //   cudaMemcpy2DAsync(final_output_data, out_length*sizeof(float), core_output_data, core_length*sizeof(float), core_length*sizeof(float), batch*out_channels, cudaMemcpyDeviceToDevice, stream);
    // But we must first zero the final_output, especially the padding columns. We can use cudaMemsetAsync to zero the entire final_output, then copy the core part.
    // That's two calls, acceptable.
    // So algorithm:
    // - Compute core_out_length = out_length - output_padding.
    // - Create core_output tensor of shape [batch, out_channels, core_out_length] and use cuDNN to compute transposed conv.
    // - Create final_output tensor of shape [batch, out_channels, out_length].
    // - cudaMemsetAsync(final_output.data_ptr<float>(), 0, total_final_bytes, stream);
    // - cudaMemcpy2DAsync(final_output.data_ptr<float>(), out_length*sizeof(float), core_output.data_ptr<float>(), core_out_length*sizeof(float), core_out_length*sizeof(float), batch*out_channels, cudaMemcpyDeviceToDevice, stream);
    // - If bias_opt exists, add bias to all positions in final_output. We can use add_bias_launcher with bias as a 1D tensor? But add_bias expects bias shape [channels, 1]. However conv bias is 1D [channels]. We can create a temporary bias tensor of shape [channels,1] by viewing? Or we can call cudnnAddTensor? Simpler: we can use the add_bias_launcher if we reshape bias to [channels,1]? But bias is 1D, but add_bias kernel expects bias pointer with contiguous elements of size channels, and it indexes bias[c] via pointer arithmetic. If bias is contiguous 1D tensor of size channels, its data_ptr is the same as if it were [channels,1]. The kernel only uses bias[c], so we can pass the 1D tensor's data_ptr directly, but add_bias wrapper expects bias tensor with shape [channels,1]. However, we can call the launcher directly instead of the wrapper. Alternatively, we can create a view of bias as [channels,1] using bias.view({channels,1}) but that's a torch operation but not a computation, it's just a view, allowed? The model_new.py uses torch operations for tensor creation and properties only, but in binding we can use torch::Tensor methods like view to get a view, I think that's allowed since it's not a computational operator but a metadata operation. However, to be safe, we can just call add_bias_launcher directly with bias data_ptr and proper strides (bias is contiguous, so bias has no batch dimension, but kernel expects bias with just channels and length 1? In add_bias_kernel, bias is indexed as bias[c]; it does not use length for bias. So we can pass bias.data_ptr<float>() and any batch/length for bias (not used). So we can call add_bias_launcher with bias_opt->data_ptr<float>() and channels=out_channels, but we need to set batch, length appropriately? Actually add_bias_launcher signature has batch, channels, length, etc., but those are for the input and output, not for bias. For bias, only channels matters, and bias pointer is assumed to be at least channels long. So we can call add_bias_launcher directly here, passing final_output as input and output (in-place), bias.data_ptr, batch, out_channels, out_length, and appropriate strides for final_output. Since final_output is contiguous, strides are straightforward. But note that add_bias_launcher writes output = input + bias; if we pass the same pointer for input and output, it will perform in-place addition, which is fine. However, we need to ensure that input and output pointers are the same and the kernel reads input and writes output; since it's element-wise, in-place is safe. So we can do:
    //   add_bias_launcher(final_output.data_ptr<float>(), bias.data_ptr<float>(), final_output.data_ptr<float>(), batch, out_channels, out_length, final_output.stride(0), final_output.stride(1), final_output.stride(2), final_output.stride(0), final_output.stride(1), final_output.stride(2), stream);
    // This will add the conv bias to all positions, including the padded positions, which are zero, so they become bias value, matching PyTorch.
    // Perfect.
    // So we can implement conv_transpose1d_forward without using torch ops except empty_like (for output allocation) and view? We'll use empty for core and final.
    // Step 1: Compute core output using cuDNN.
    // Step 2: Allocate final output and zero it.
    // Step 3: Copy core into final using cudaMemcpy2DAsync.
    // Step 4: If bias exists, add bias in-place.
    // Step 5: Return final output.

    // However, we also need to handle input contiguity: cuDNN requires contiguous input? PyTorch may pass non-contiguous input if it has strides, but contract says input layout "supports strides". So we may need to make input contiguous before calling cuDNN. We can check input.is_contiguous(), and if not, call input.contiguous() to get a contiguous copy. That uses torch operation but it's just memory layout change, allowed. We'll do that.
    // Similarly, weight is required contiguous, we'll ensure.
    // Bias can be non-contiguous? contract says bias contiguous, but we'll ensure contiguity for safety.

    // cuDNN transposed convolution: use cudnnConvolutionBackwardData.
    // We need to set descriptors: input_desc for gradInput? Actually for backward data, the "input" to the operation is the output gradient, and the "output" is the input gradient. But we can rename: dy = input tensor (our input), w = weight, dx = output tensor (our core_output). So we set dy_desc = input_desc (our input), w_desc = filter_desc, dx_desc = output_desc for core length. Then call cudnnConvolutionBackwardData.
    // We need to specify alpha, beta, workspace.
    // We'll allocate output tensor (core_output) of appropriate size.
    // Workspace: need to get workspace size via cudnnGetConvolutionBackwardDataWorkspaceSize. We'll allocate via cudaMalloc.
    // Then call cudnnConvolutionBackwardData.
    // After that, we can destroy descriptors and handle.

    // Implementation details:
    // - Convert batch, channels, lengths to int for cuDNN (they are int).
    // - Set input_desc with our input tensor dimensions (N, C, H=1, W=in_length).
    // - Set filter_desc with weight layout: PyTorch stores weight for ConvTranspose1d as [in_channels, out_channels, kernel_size]. cuDNN expects filter as [out_channels, in_channels/groups, kernel_h, kernel_w] for backward data? Actually for cudnnConvolutionBackwardData, the filter descriptor is same as forward convolution, which is [out_channels, in_channels, kernel_h, kernel_w]. But PyTorch's ConvTranspose weight is the weight of the corresponding forward convolution, so it is [in_channels, out_channels, kernel_size] (transposed of forward weight). So to use cuDNN, we need to treat the weight as the forward weight and use cudnnConvolutionBackwardData, which will compute dx from dy using the same filter descriptor as forward. However, for backward data, the filter is the same as forward (list of kernels for each output channel). So we need to set filter_desc with dimensions: [out_channels, in_channels, kernel_h, kernel_w] = [weight.size(1), weight.size(0), 1, weight.size(2)]. But we cannot use weight directly as it is stored in PyTorch layout [in_channels, out_channels, kernel_size]; that is transposed relative to cuDNN's expectation. Actually cuDNN's filter descriptor for convolution is [K, C, R, S] where K = number of output feature maps, C = number of input feature maps. PyTorch's Conv2d weight is [out_channels, in_channels, kh, kw], i.e., [K, C, R, S]. That matches cuDNN. For ConvTranspose2d, PyTorch's weight is [in_channels, out_channels, kh, kw], which is the transposed version. But when using cudnnConvolutionBackwardData to implement transposed conv, we need the filter as if for the forward convolution, i.e., [out_channels, in_channels, kh, kw]. Since PyTorch stores the transposed weight [in_channels, out_channels, kh, kw], we need to pass the filter descriptor as [out_channels, in_channels, kh, kw] with appropriate strides to point to the transposed layout? That would require a custom filter format. Alternatively, we can use cudnnSetFilter4dDescriptor with format CUDNN_TENSOR_NCHW and dimensions K=weight.size(1), C=weight.size(0), R=1, S=weight.size(2), and provide the weight pointer directly. But the actual memory layout of weight is [in_channels, out_channels, kernel_size]. cuDNN expects the filter to be laid out as [K, C, R, S] contiguous in that order. However, if we tell cuDNN that K=weight.size(1), C=weight.size(0), R=1, S=weight.size(2), the total number of elements is same, but the ordering is different: cuDNN expects first K outer loops, then C, then R, then S. PyTorch's weight has C outer loops (in_channels), then K (out_channels), then S. So the memory layout does not match the descriptor. To use it, we would need to transpose the weight, which requires a copy and possible transpose kernel. Since we cannot add a transpose kernel, we might need to use torch::transpose? That's a tensor operation and might be considered a torch op? It's a metadata operation but may allocate a new tensor with different strides, but the data remains unchanged. cuDNN expects a contiguous filter in the specified format. If we pass weight with strides that correspond to the transposed dimensions, cuDNN can read it if we set the filter descriptor with appropriate strides? cudnnSetFilter4dDescriptor does not allow strides; it assumes contiguous layout. There is cudnnSetFilter4dDescriptor_v4 which allows strides, but may be deprecated. Perhaps we can use cudnnSetFilter4dDescriptor with format CUDNN_TENSOR_NCHW and then use cudnnSetFilter4dDescriptor with a custom stride via cudnnSetFilter4dDescriptor_v4? This is getting too complex.
    // A simpler approach: Since we are allowed to use cuBLAS and cuDNN, but we can also use torch operations for memory manipulation? The restriction says "NO torch operators in C++: NEVER use torch::* or torch::nn::functional::* in binding.cpp or .cu files". This likely means we cannot use computational torch operators, but tensor methods like .transpose(), .contiguous() are part of the tensor API and not computational operators. However, to be safe, we should avoid any torch:: operation that performs computation. View, transpose, contiguous are not computational but can allocate or reorder memory. contiguous may copy data, but it's still not a math operation. I think it's acceptable to use tensor.view, tensor.transpose, tensor.contiguous for layout adjustments as they are not performing mathematical computation. But the instruction is strict: "NEVER use torch::* or torch::nn::functional::*". `torch::Tensor::transpose` is a method, not a torch:: function. However, the spirit is to avoid using PyTorch's high-level ops for computation. I'll assume we can use tensor methods that only manipulate metadata or copy if necessary, like .contiguous(), .view(), .permute(). But we should minimize usage.
    // For weight transposition: PyTorch stores ConvTranspose1d weight as [in_channels, out_channels, K]. To use cuDNN, we can either:
    //   a) Transpose weight to [out_channels, in_channels, K] in a temporary tensor using weight.permute({1,0,2}).contiguous(), then pass that to cuDNN. This uses torch operations but only memory layout, no math. We'll do that.
    //   b) Use cuDNN's backward filter? No.
    // We'll go with (a).

    // So we'll create a transposed weight tensor: auto weight_t = weight.permute({1,0,2}).contiguous();
    // Then set filter descriptor with dimensions weight_t.size(0)=out_channels, weight_t.size(1)=in_channels, kernel_h=1, kernel_w=kernel_size.
    // Use weight_t.data_ptr<float>().

    // Additionally, input may be non-contiguous; we'll make it contiguous: auto input_contig = input.contiguous();
    // Output core tensor will be contiguous.

    // Now, we need to set output descriptor for core length (without output_padding).
    // We'll create tensor core_output = torch::empty({batch, out_channels, core_out_length}, input.options());
    // Then call cudnnConvolutionBackwardData with alpha=1, beta=0.
    // After that, allocate final_output = torch::empty({batch, out_channels, out_length}, input.options());
    // Then zero final_output: cudaMemsetAsync(final_output.data_ptr<float>(), 0, final_output.numel()*sizeof(float), stream);
    // Copy core into final with cudaMemcpy2DAsync.
    // Then add bias if present using add_bias_launcher directly.

    // We need to ensure all cuDNN calls are checked and descriptors destroyed.

    // Let's implement this step by step.

    // Clean up descriptors and handle in a scope.

    // Note: We must also handle groups (always 1 here).

    // Let's write the code.
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("add_bias", &add_bias, "Add bias with broadcast",
          py::arg("input"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
