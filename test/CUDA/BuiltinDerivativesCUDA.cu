// clad ships a pullback for rsqrtf, __logf and __expf, and all three are unary.
// A unary call resolves to a pushforward in both directions, so nothing could
// reach those three: reaching one was an error rather than a fallback, and the
// error was reported against the caller's own line as a "user-defined
// derivative" that clad ships itself. See #2172.

// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -o%t -Xclang -verify %s \
// RUN:     2>&1 | %filecheck %s
//
// REQUIRES: cuda-compile
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"

__device__ float intrinsics(float a) {
  return rsqrtf(a) + __logf(a) + __expf(a);
}

__global__ void kernel(float* out, const float* in) {
  *out = intrinsics(*in);
}

int main() {
  clad::gradient(kernel);
}

// The three unary intrinsics have to reach the pushforwards clad now ships.
// CHECK-DAG: rsqrtf_pushforward
// CHECK-DAG: __logf_pushforward
// CHECK-DAG: __expf_pushforward
