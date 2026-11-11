#include <cuda_runtime.h>

// No custom CUDA kernels are required for this model.
// Conv1d is implemented via cuDNN in the corresponding C++ binding file.
// MaxPool1d is also implemented via cuDNN or native PyTorch ops in the binding file.
// This file exists to satisfy the requirement of having a single .cu file in kernels/.
