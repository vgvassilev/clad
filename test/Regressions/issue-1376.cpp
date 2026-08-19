// RUN: %cladclang -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

namespace clad::custom_derivatives::overrides::std {
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> exp_pushforward(T x, dT d_x) {
  return {::std::exp(x), 2 * ::std::exp(x) * d_x};
}
}
double differentiable_code(double x) {
    return std::exp(x);
}

// CHECK: void differentiable_code_grad(double x, double *_d_x) {
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         _r0 += 1 * clad::custom_derivatives::overrides::std::exp_pushforward(x, 1.).pushforward;
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
  auto df = clad::gradient(differentiable_code);
  df.dump();
  double x = 1.0, d_x = 0;
  df.execute(x, &d_x);
  printf("d_x = %.5f\n", d_x);
  // CHECK-EXEC: d_x = 5.43656
}
