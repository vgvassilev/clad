// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A field can be named as the independent variable of a parameter passed by
// reference, just as of one passed by value: the parameter's type is then a
// reference to the class, and the class is what the fields are looked up in.

#include "clad/Differentiator/Differentiator.h"

struct T {
  double x;
};

struct Outer {
  T t;
};

double byValue(T t) { return t.x * t.x; }

double byRef(T& t) { return t.x * t.x; }

// CHECK: double byRef_darg0_x(T &t) {
// CHECK-NEXT:     T _d_t = {0.};
// CHECK-NEXT:     _d_t.x = 1;
// CHECK-NEXT:     double &_t0 = t.x;
// CHECK-NEXT:     double &_t1 = t.x;
// CHECK-NEXT:     return _d_t.x * _t1 + _t0 * _d_t.x;
// CHECK-NEXT: }

double nestedByRef(Outer& o) { return o.t.x * o.t.x; }

// CHECK: double nestedByRef_darg0_t_x(Outer &o) {
// CHECK-NEXT:     Outer _d_o = {{[{][{]}}0.{{[}][}]}};
// CHECK-NEXT:     _d_o.t.x = 1;

int main() {
  T t{3};
  Outer o{{5}};

  auto dByValue = clad::differentiate(byValue, "t.x");
  printf("%.2f\n", dByValue.execute(t)); // CHECK-EXEC: 6.00

  auto dByRef = clad::differentiate(byRef, "t.x");
  printf("%.2f\n", dByRef.execute(t)); // CHECK-EXEC: 6.00

  auto dNested = clad::differentiate(nestedByRef, "o.t.x");
  printf("%.2f\n", dNested.execute(o)); // CHECK-EXEC: 10.00

  auto gByRef = clad::gradient(byRef, "t.x");
  T dt{0};
  gByRef.execute(t, &dt);
  printf("%.2f\n", dt.x); // CHECK-EXEC: 6.00
}
