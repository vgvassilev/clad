// RUN: %cladclang %s -I%S/../../include -oSideEffectArgs.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./SideEffectArgs.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify \
// RUN:   -DUNSUPPORTED

// The call, the stores around it and its pullback each use an argument, but
// an argument that writes something, such as `a[i++]`, must be evaluated
// once. clad warns about one it cannot keep from before its writes.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct S {
  double a;
};

void scale(S& s, double k) { s.a *= k; }

// Passed by reference: keep the addresses of the element and of its adjoint,
// then step the index, once.
double byReference(double x, int n) {
  S arr[4] = {{x}, {x}, {x}, {x}};
  int i = 0;
  for (int k = 0; k < n; ++k)
    scale(arr[i++], x);
  scale(arr[++i], x);
  return arr[0].a + arr[1].a + arr[3].a;
}

// CHECK: void byReference_grad_0(double x, int n, double *_d_x) {
// CHECK:     clad::push(_t1, clad_addressof(arr[i]));
// CHECK-NEXT:     clad::push(_t2, clad_addressof(_d_arr[i]));
// CHECK-NEXT:     i++;
// CHECK:     {{scale|scale_reverse_forw}}(*clad::back(_t1), x
// CHECK:     S *_t4 = clad_addressof(arr[(i + 1)]);
// CHECK-NEXT:     S *_t5 = clad_addressof(_d_arr[(i + 1)]);
// CHECK-NEXT:     ++i;

// An assignment is a write too: `j += 1` runs once.
double byAssignment(double x) {
  S arr[2] = {{x}, {x}};
  int j = 0;
  scale(arr[j += 1], x);
  scale(arr[j = 0], x);
  return arr[0].a + arr[1].a * (j + 1);
}

// CHECK: void byAssignment_grad(double x, double *_d_x) {
// CHECK:     S *_t1 = clad_addressof(arr[1 + j]);
// CHECK-NEXT:     S *_t2 = clad_addressof(_d_arr[1 + j]);
// CHECK-NEXT:     j += 1;
// CHECK:     S *_t5 = clad_addressof(arr[0]);
// CHECK-NEXT:     S *_t6 = clad_addressof(_d_arr[0]);
// CHECK-NEXT:     j = 0;

// A member and an index that writes nothing on the way are fine too.
struct Grid {
  S rows[2][2];
};

double byMember(double x) {
  Grid g{{{{x}, {x}}, {{x}, {x}}}};
  int i = 0;
  scale(g.rows[1][i++], x);
  return g.rows[1][0].a * i;
}

// CHECK: void byMember_grad(double x, double *_d_x) {
// CHECK:     S *_t{{[0-9]+}} = clad_addressof(g.rows[1][i]);
// CHECK-NEXT:     S *_t{{[0-9]+}} = clad_addressof(_d_g.rows[1][i]);
// CHECK-NEXT:     i++;

// A call reverse mode differentiates with a pushforward uses its argument
// twice too.
double sq(double v) { return v * v; }

double byPushforward(double x) {
  int i = 0;
  double r = sq(x + i++);
  return r + i;
}

// CHECK: void byPushforward_grad(double x, double *_d_x) {
// CHECK:     double _t{{[0-9]+}} = x + i++;

// Passed by value: keep the value.
double byValue(double x) {
  S s{x};
  int i = 0;
  double r = s.a * x;
  scale(s, x + i++);
  return r * i;
}

// CHECK: void byValue_grad(double x, double *_d_x) {
// CHECK:     double _t1 = x + i++;
// CHECK:     {{scale|scale_reverse_forw}}(s, _t1
// CHECK:     scale_pullback(s, _t1, &_d_s, &_r0);

#ifdef UNSUPPORTED
// Reverse mode cannot keep the element from before `p++`: clad says so.
double byPointer(double x) {
  S arr[2] = {{x}, {x}};
  S* p = arr;
  scale(*p++, x); // expected-warning {{clad evaluates this argument more than once, and repeats what it writes each time}}
  return arr[0].a;
}

void unsupported() { clad::gradient(byPointer); }
#endif

int main() {
  double dx = 0;
  // The elements 0, 1 and 3 become x^2 each: 3 x^2.
  clad::gradient(byReference, "x").execute(3, 2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 18.00
  dx = 0;
  clad::gradient(byAssignment).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 12.00
  dx = 0;
  clad::gradient(byMember).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(byPushforward).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(byValue).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
}
