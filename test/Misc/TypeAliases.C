// RUN: %cladclang %s -I%S/../../include -oTypeAliases.out 2>&1 | %filecheck %s
// RUN: ./TypeAliases.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oTypeAliases.out
// RUN: ./TypeAliases.out | %filecheck_exec %s

// An alias is where portable code picks a type -- a precision, a width, an
// index -- and the code around it is written in those terms. clad carries the
// alias into the derivative and keeps it as written, so the derivative stays
// portable in the same way the primal was.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

template <typename T> struct Traits {
  using value_type = T;
  typedef unsigned long index_type;
};

// Reverse mode promotes the declarations it derives to function scope, so an
// alias they are written in terms of has to move there with them.
template <typename T> T weighted(const T* x, int n) {
  using elem = typename Traits<T>::value_type;
  typedef typename Traits<T>::index_type index;
  elem s = 0;
  for (index i = 0; i < (index)n; ++i) {
    elem t = x[i] * x[i];
    s += t;
  }
  return s;
}
// CHECK: void weighted_grad_0(const double *x, int n, double *_d_x) {
// CHECK-NEXT: int _d_n = 0;
// CHECK-NEXT: using elem = typename Traits<double>::value_type;
// CHECK-NEXT: typedef typename Traits<double>::index_type index;
// CHECK-NEXT: index _d_i = 0UL;
// CHECK-NEXT: index i = 0UL;
// CHECK-NEXT: elem _d_t = 0.;
// CHECK-NEXT: elem t = 0.;
// CHECK-NEXT: elem _d_s = 0.;
// CHECK-NEXT: elem s = 0;

// The alias sits in the loop the promoted declarations come out of, so it has
// to leave with them rather than stay behind.
double innerAlias(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i) {
    using elem = double;
    elem t = x[i] * x[i];
    s += t;
  }
  return s;
}
// CHECK: void innerAlias_grad_0(const double *x, int n, double *_d_x) {
// CHECK: using elem = double;
// CHECK-NEXT: elem _d_t = 0.;
// CHECK-NEXT: elem t = 0.;

// Forward mode leaves declarations where it finds them, and an alias built on
// an earlier alias still reads.
double pointerAlias(double x) {
  typedef double real;
  using ptr = const real*;
  real y = x * x;
  ptr p = &y;
  return *p * x;
}
// CHECK: double pointerAlias_darg0(double x) {
// CHECK-NEXT: double _d_x = 1;
// CHECK-NEXT: typedef double real;
// CHECK-NEXT: using ptr = const real *;
// CHECK-NEXT: real _d_y = _d_x * x + x * _d_x;
// CHECK-NEXT: real y = x * x;

int main() {
  double x[4] = {1, 2, 3, 4};
  double dx[4] = {0, 0, 0, 0};

  auto gw = clad::gradient(weighted<double>, "x");
  gw.execute(x, 4, dx);
  printf("weighted: {%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]);
  // CHECK-EXEC: weighted: {2.00, 4.00, 6.00, 8.00}

  for (int i = 0; i < 4; ++i)
    dx[i] = 0;
  auto gi = clad::gradient(innerAlias, "x");
  gi.execute(x, 4, dx);
  printf("innerAlias: {%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]);
  // CHECK-EXEC: innerAlias: {2.00, 4.00, 6.00, 8.00}

  auto dp = clad::differentiate(pointerAlias, "x");
  printf("pointerAlias: %.2f\n", dp.execute(3));
  // CHECK-EXEC: pointerAlias: 27.00
}
