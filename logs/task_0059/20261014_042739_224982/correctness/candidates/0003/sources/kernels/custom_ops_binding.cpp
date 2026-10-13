// 新实现：不使用 cuDNN，改用 cuBLAS GEMM 实现 conv_transpose2d，或自定义 kernel。
// 此处先采用自定义 CUDA kernel 方案，避免 cuBLAS 的复杂转换。
// 注意：转置卷积可以通过对输入进行填充后执行常规卷积的转置来实现，也可以使用 col2im 方式。
// 为了简单和正确，我们使用 im2col + GEMM 方法：
// 将 conv_transpose2d 视为输入与翻转权重之间的常规卷积的转置，等价于对输入进行零填充（根据 stride 和 padding），然后与旋转180度的权重做常规卷积。
// 但常规卷积需要 cuDNN 不可用，所以我们也需要自定义卷积 kernel 或使用 cuBLAS。
// 最终决定：在 custom_ops.cu 中实现一个通用的 conv_transpose2d kernel，该 kernel 不依赖 cuBLAS 或 cuDNN。
// binding.cpp 只做张量检查和 launch。
