// The only demo here that differentiates a function launching a kernel of
// its own rather than host code driving Thrust.
//
// RUN: %cladclang_cuda -I%S/../../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -o%t \
// RUN:     %S/../../../demos/CUDA/TensorContraction.cu 2>&1 | %filecheck_nodiag %s
//
// Running these needs a device, and what they print is a list of
// gradients rather than anything stable to match, so compiling them
// is the check.
//
// REQUIRES: cuda-compile
