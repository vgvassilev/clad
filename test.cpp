// RUN: %cladnumdiffclang -O2 -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -std=c++17 -I%S/../../include -oGradients.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./Gradients.out
// RUN: ./Gradients.out | %filecheck_exec %s
// RUN: %cladnumdiffclang %s  -I%S/../../include -oGradients.out
// RUN: ./Gradients.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cstdio>

#include "../TestUtils.h"


template<size_t N>
double fn_template_non_type(double x) {
  const size_t maxN = 53;
  const size_t m = maxN < N ? maxN : N;
  return x*m;
}

// CHECK: template<> void fn_template_non_type_grad<{{15ULL|15UL|15U|15}}>(double x, double *_d_x) {
// CHECK-NEXT:     size_t _d_maxN = {{0U|0UL}};
// CHECK-NEXT:     const size_t maxN = 53;
// CHECK-NEXT:     bool _cond0 = maxN < {{15U|15UL|15ULL}};
// CHECK-NEXT:     size_t _d_m = {{0U|0UL}};
// CHECK-NEXT:     const size_t m = _cond0 ? maxN : {{15U|15UL|15ULL}};
// CHECK-NEXT:     {
// CHECK-NEXT:       *_d_x += 1 * m;
// CHECK-NEXT:       _d_m += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     if (_cond0)
// CHECK-NEXT:         _d_maxN += _d_m;
// CHECK-NEXT: }

int main() {
  auto fn_template_non_type_dx = clad::gradient(fn_template_non_type<15>);
  double x = 5, dx = 0;
  fn_template_non_type_dx.execute(x, &dx);
  printf("Result is = %.2f\n", dx); // CHECK-EXEC: Result is = 15.00
}
