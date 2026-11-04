// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A member read through a pointer without a tangent has none either: a scalar
// reads as 0 and anything else stays without one.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct S {
  double v;
  double arr[2];
};
S s{2, {3, 4}};
S* gp = &s;

// CHECK: warning: 'v' is accessed through an expression without a derivative; its derivative is treated as 0
double product(double x) { return gp->v * x; }

// CHECK: warning: 'arr' is accessed through an expression without a derivative; its derivative is treated as 0
double element(double x) { return gp->arr[1] * x; }

// A class-typed member gets no tangent, not a 0 whose address a member call
// would take.
struct Box {
  S* p;
  S& operator*() const { return *p; }
};
struct Holder { Box box; };
Holder h{{&s}};
Holder* hp = &h;

// CHECK: warning: 'box' is accessed through an expression without a derivative; its derivative is treated as 0
double through_box(double x) {
  auto& r = *hp->box;
  return r.v * x;
}

// CHECK: double product_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double &_t0 = gp->v;
// CHECK-NEXT:     return _t0 * _d_x;
// CHECK-NEXT: }

int main() {
  printf("%.2f\n", clad::differentiate(product, "x").execute(3));     // CHECK-EXEC: 2.00
  printf("%.2f\n", clad::differentiate(element, "x").execute(3));     // CHECK-EXEC: 4.00
  printf("%.2f\n", clad::differentiate(through_box, "x").execute(3)); // CHECK-EXEC: 2.00
}
