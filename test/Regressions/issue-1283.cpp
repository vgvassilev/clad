// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

struct Struct {
  double val;
};

double fn(double a) {
  double result = 0;
  for (int i = 0; i < 3; ++i) {
    Struct s;
    s.val = a * (i + 1);
    result += s.val;
  }
  return result;
}

int main() {
  auto grad = clad::gradient(fn);
  double d_a = 0;
  grad.execute(3, &d_a);
  printf("%.2f\n", d_a); // CHECK-EXEC: 6.00
}

// CHECK: void fn_grad(double a, double *_d_a) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     Struct _d_s = {0.};
// CHECK-NEXT:     Struct s = {0.};
// CHECK-NEXT:     double _d_result = 0.;
// CHECK-NEXT:     double result = 0;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < 3; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         s = {0.};
// CHECK-NEXT:         s.val = a * (i + 1);
// CHECK-NEXT:         result += s.val;
