// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct A {
  double val;
};

A objVal(double x) {
  return {x * x};
}

double f(double x, double y) {
  return objVal(x).val;
}

double g(double x) {
  return 5.0 * objVal(x).val;
}

struct B {
  double val;
  ~B() {}
};

B makeB(double x) {
  return {x * x};
}

double h(double x) {
  return makeB(x).val;
}

// CHECK: void objVal_pullback(double x, A _d_y, double *_d_x)
// CHECK: void f_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK:   A _r{{[0-9]+}} =
// CHECK:   _r{{[0-9]+}}.val +=
// CHECK:   objVal_pullback(x,
// CHECK:   *_d_x +=
// CHECK: }

// CHECK: void g_grad(double x, double *_d_x) {
// CHECK:   A _r{{[0-9]+}} =
// CHECK:   _r{{[0-9]+}}.val +=
// CHECK:   objVal_pullback(x,
// CHECK:   *_d_x +=
// CHECK: }

// CHECK: void makeB_pullback(double x, B _d_y, double *_d_x)

// CHECK: void h_grad(double x, double *_d_x) {
// CHECK:   B _r{{[0-9]+}} =
// CHECK:   _r{{[0-9]+}}.val +=
// CHECK:   makeB_pullback(x,
// CHECK:   *_d_x +=
// CHECK: }

int main() {
  auto f_grad = clad::gradient(f);
  double dx = 0, dy = 0;
  f_grad.execute(3.0, 4.0, &dx, &dy);
  printf("dx=%.1f dy=%.1f\n", dx, dy);
  // CHECK-EXEC: dx=6.0 dy=0.0

  auto g_grad = clad::gradient(g);
  dx = 0;
  g_grad.execute(3.0, &dx);
  printf("gdx=%.1f\n", dx);
  // CHECK-EXEC: gdx=30.0

  auto h_grad = clad::gradient(h);
  dx = 0;
  h_grad.execute(3.0, &dx);
  printf("hdx=%.1f\n", dx);
  // CHECK-EXEC: hdx=6.0
}