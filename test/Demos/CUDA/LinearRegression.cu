// The squared error of a linear model, differentiated with respect to its
// weights. thrust::inner_product is the operation clad has to know a
// derivative for.
//
// RUN: %cladclang_cuda -I%S/../../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -o%t \
// RUN:     %S/../../../demos/CUDA/LinearRegression.cu 2>&1 | %filecheck_nodiag %s
//
// Running these needs a device, and what they print is a list of
// gradients rather than anything stable to match, so compiling them
// is the check.
//
// REQUIRES: cuda-compile
