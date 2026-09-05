// RUN: %cladclang %s -I%S/../../include -oPointerAllocation.out
// RUN: ./PointerAllocation.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double allocThenAssign(double x) {
  double* p = nullptr;
  p = new double[2];
  double* owner = p;
  p[0] = x * x;
  double r = p[0];
  delete[] owner;
  return r;
}

// CHECK: void allocThenAssign_grad(double x, double *_d_x) {
// CHECK-NEXT:  double *_d_p = nullptr;
// CHECK-NEXT:  double *p = nullptr;
// CHECK-NEXT:  double *_t0 = _d_p;
// CHECK-NEXT:  _d_p = new double [2](/*implicit*/(double[2])0);
// CHECK-NEXT:  p = new double [2];
// CHECK-NEXT:  double *_d_owner = _d_p;
// CHECK-NEXT:  double *owner = p;
// CHECK-NEXT:  p[0] = x * x;
// CHECK-NEXT:  double _d_r = 0.;
// CHECK-NEXT:  double r = p[0];
// CHECK-NEXT:  _d_r += 1;
// CHECK-NEXT:  _d_p[0] += _d_r;
// CHECK-NEXT:  {
// CHECK-NEXT:    double _r_d0 = _d_p[0];
// CHECK-NEXT:    _d_p[0] = 0.;
// CHECK-NEXT:    *_d_x += _r_d0 * x;
// CHECK-NEXT:    *_d_x += x * _r_d0;
// CHECK-NEXT:  }
// CHECK-NEXT:  _d_p = _t0;
// CHECK-NEXT:  delete [] owner;
// CHECK-NEXT:  delete [] _d_owner;
// CHECK-NEXT:  }

int main() {
  double dx = 0;
  auto grad=clad::gradient(allocThenAssign);
  grad.execute(3, &dx);
  printf("{%.2f}\n", dx); // CHECK-EXEC: {6.00}
}
