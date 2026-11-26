// RUN: %cladclang -std=c++17 -I%S/../../include %s -o%t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// Regression test for issue #1485:
// Deduced `auto` variables of namespaced types must not repeat their namespace
// qualifier when declared in generated derivative code.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

namespace cladtorch {
struct Tensor {
  float data;
};
} // namespace cladtorch

namespace outer {
namespace inner {
struct Vec {
  double x;
};
} // namespace inner
} // namespace outer

float fn1(const cladtorch::Tensor& t) {
  auto b = t;
  return b.data;
}

// CHECK: void fn1_grad(const cladtorch::Tensor &t, cladtorch::Tensor *_d_t) {
// CHECK-NEXT: cladtorch::Tensor _d_b = (*_d_t);
// CHECK-NEXT: cladtorch::Tensor b = t;
// CHECK-NEXT: _d_b.data += 1;
// CHECK-NEXT: cladtorch::Tensor::constructor_pullback(t, &_d_b, _d_t);
// CHECK-NEXT: }

float fn_ref_ptr(cladtorch::Tensor* t) {
  const auto& r = *t;
  auto* p = t;
  return r.data + p->data;
}

// CHECK: void fn_ref_ptr_grad(cladtorch::Tensor *t, cladtorch::Tensor *_d_t) {
// CHECK-NEXT: cladtorch::Tensor &_d_r = *_d_t;
// CHECK-NEXT: const cladtorch::Tensor &r = *t;
// CHECK-NEXT: cladtorch::Tensor *_d_p = _d_t;
// CHECK-NEXT: cladtorch::Tensor *p = t;

double fn_nested(const outer::inner::Vec& v) {
  auto copy = v;
  return copy.x * 2.0;
}

// CHECK: void fn_nested_grad(const outer::inner::Vec &v, outer::inner::Vec *_d_v) {
// CHECK-NEXT: outer::inner::Vec _d_copy = (*_d_v);
// CHECK-NEXT: outer::inner::Vec copy = v;

float fn_fwd(cladtorch::Tensor t, float x) {
  auto b = t;
  return b.data * x;
}

// CHECK: float fn_fwd_darg1(cladtorch::Tensor t, float x) {
// CHECK-NEXT: cladtorch::Tensor _d_t = {0.F};
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: cladtorch::Tensor _d_b = _d_t;
// CHECK-NEXT: cladtorch::Tensor b = t;

int main() {
  cladtorch::Tensor t{3.0F};
  cladtorch::Tensor d_t{0.0F};
  clad::gradient(fn1).execute(t, &d_t);
  printf("fn1 %.2f\n", d_t.data); // CHECK-EXEC: fn1 1.00

  cladtorch::Tensor d_t2{0.0F};
  clad::gradient(fn_ref_ptr).execute(&t, &d_t2);
  printf("fn_ref_ptr %.2f\n", d_t2.data); // CHECK-EXEC: fn_ref_ptr 2.00

  outer::inner::Vec v{4.0};
  outer::inner::Vec d_v{0.0};
  clad::gradient(fn_nested).execute(v, &d_v);
  printf("fn_nested %.2f\n", d_v.x); // CHECK-EXEC: fn_nested 2.00

  printf("fn_fwd %.2f\n", clad::differentiate(fn_fwd, "x").execute(t, 5.0F));
  // CHECK-EXEC: fn_fwd 3.00
  return 0;
}
