// RUN: %cladclang -DCROSSTU_SECOND_TU -O0 -c %s -I%S/../../include -o %t-tu2.o
// RUN: %cladclang -O0 %s %t-tu2.o -I%S/../../include -o %t.out 2>&1 | %filecheck %s
// RUN: %t.out | %filecheck_exec %s

// Two translation units, this file built twice, take the gradient of the same
// inline function and still link: the root derivative follows the primal's
// inline-ness, so both copies are linkonce_odr. At -O0 nothing is inlined
// away and both copies reach the linker.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

inline double g(double x, double y) { return x * x * y + y; }

#ifdef CROSSTU_SECOND_TU

void grad_in_tu2(double x, double y, double* dx, double* dy) {
  clad::gradient(g).execute(x, y, dx, dy);
}

#else

void grad_in_tu2(double x, double y, double* dx, double* dy);

// CHECK: inline void g_grad(double x, double y, double *_d_x, double *_d_y) {

int main() {
  double dx = 0, dy = 0;
  clad::gradient(g).execute(3, 4, &dx, &dy);
  printf("tu1: {%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: tu1: {24.00, 10.00}
  dx = dy = 0;
  grad_in_tu2(3, 4, &dx, &dy);
  printf("tu2: {%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: tu2: {24.00, 10.00}
}

#endif
