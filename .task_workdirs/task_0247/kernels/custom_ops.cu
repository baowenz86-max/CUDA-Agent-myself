// CUDA kernels for custom ops.
// This model only uses ConvTranspose1d, Conv1d and bias broadcast addition.
// Conv/ConvTranspose must be implemented via cuDNN in the C++ binding layer.
// No non-convolution CUDA kernels are required in this task.
// This file intentionally contains no kernel definitions.
#include <cuda_runtime.h>
