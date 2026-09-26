// Logistic regression over a bag of words, one sample at a time and then
// a batch at a time. Both the per-sample and the batched loss are
// differentiated.
//
// RUN: %cladclang_cuda -I%S/../../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -o%t \
// RUN:     %S/../../../demos/CUDA/BoWLogisticRegression.cu 2>&1 | %filecheck_nodiag %s
//
// Running these needs a device, and what they print is a list of
// gradients rather than anything stable to match, so compiling them
// is the check.
//
// REQUIRES: cuda-compile
