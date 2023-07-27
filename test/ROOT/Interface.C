// RUN: %cladclang %s -I%S/../../include -oInterface.out 2>&1 | FileCheck %s
// RUN: ./Interface.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

using Double_t = double;

// This struct is defined in ROOT TFormula.h file as a lightweight data
// structure for array_ref
struct array_ref_interface {
  Double_t *arr;
  std::size_t size;
};

Double_t f(Double_t* x, Double_t* p) {
  return p[0] + x[0] * p[1];
}

void f_grad_1(Double_t* x, Double_t* p, clad::array_ref<Double_t> _d_p);

// CHECK: void f_grad_1(Double_t *x, Double_t *p, clad::array_ref<Double_t> _d_p) {
// CHECK-NEXT:     Double_t _t0;
// CHECK-NEXT:     Double_t _t1;
// CHECK-NEXT:     _t1 = x[0];
// CHECK-NEXT:     _t0 = p[1];
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_p[0] += 1;
// CHECK-NEXT:         {{double|Double_t}} _r0 = 1 * _t0;
// CHECK-NEXT:         {{double|Double_t}} _r1 = _t1 * 1;
// CHECK-NEXT:         _d_p[1] += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
  Double_t x[] = { 2 };
  Double_t p[] = { 2, 3 };
  Double_t result[2] = { 0 };

  clad::gradient(f, "p");

  // We create a struct of "array_ref_interface" type and store its address in
  // a void pointer. When the grad function is called this void pointer is
  // type casted to clad::array_ref to create a functionality that is similar
  // to reinterpret_cast.
  array_ref_interface ari = array_ref_interface{result, 2};
  void *arg = &ari;
  f_grad_1(x, p, *(clad::array_ref<Double_t> *)arg);

  printf("Result is = {%.2f, %.2f}\n", result[0], result[1]); // CHECK-EXEC: Result is = {1.00, 2.00}
}
