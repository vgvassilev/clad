// Reverse mode asks a function of one argument for its pushforward, not for
// its pullback: the reverse of a unary call is a multiplication by f'(x), so
// there is nothing left over for a separate pullback to do. The CUDA
// derivatives clad ships for __expf, __logf and rsqrtf were pullbacks with no
// pushforward beside them, so nothing could ever call them and differentiating
// a call to one was a hard error reported against the caller's own line.
// Fixes #2172.
//
// Compiling is the whole assertion here: before the pushforwards landed, every
// one of these three calls was an error rather than a fallback, so a clean
// compile under -verify is what distinguishes the two behaviours. The CHECKs
// then name the pushforwards clad picked, to show it found them rather than
// falling back to numerical differentiation. There is deliberately no execution
// half: the value each pushforward returns is the primal the demo already
// checks against its CPU counterpart, and asserting floats a device run has
// not produced yet would only be a guess.
//
// RUN: %cladclang_cuda -I%S/../../include -fsyntax-only \
// RUN:     --cuda-gpu-arch=%cudaarch --cuda-path=%cudapath -Xclang -verify \
// RUN:     %s 2>&1 | %filecheck %s
//
// REQUIRES: cuda-compile
//
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"

// __expf, __logf and rsqrtf are one-argument, so each needs a pushforward.
// CHECK-DAG: __expf_pushforward(
// CHECK-DAG: __logf_pushforward(
// CHECK-DAG: rsqrtf_pushforward(

__device__ float unaryIntrinsics(float x) {
  return __expf(x) + __logf(x) + rsqrtf(x);
}

__global__ void differentiate(float* out) {
  auto grad = clad::gradient(unaryIntrinsics);
  grad.execute(1.0F, out);
}
