// NVIDIA's BlackScholes sample with a clad gradient added, keeping the
// original CPU version beside it to check the gradients against. It carries
// NVIDIA's helper headers, which is why this one needs an include path the
// others do not.
//
// RUN: %cladclang_cuda -I%S/../../../include --cuda-path=%cudapath \
// RUN:     -I%S/../../../demos/CUDA/BlackScholes/helper \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -o%t \
// RUN:     %S/../../../demos/CUDA/BlackScholes/BlackScholes.cu 2>&1 \
// RUN:     | %filecheck_nodiag %s
//
// REQUIRES: cuda-compile
