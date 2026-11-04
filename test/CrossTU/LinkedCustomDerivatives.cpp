// RUN: %cladclang -DCROSSTU_DEFINITIONS -c %s -I%S/../../include -o %t-defs.o
// RUN: %cladclang %s %t-defs.o -I%S/../../include -o %t.out 2>&1 | %filecheck %s
// RUN: %t.out | %filecheck_exec %s

// The custom pullback is only declared here and defined in the other
// translation unit, this file built with -DCROSSTU_DEFINITIONS.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double kick(double p, double q);

namespace clad {
namespace custom_derivatives {
void kick_pullback(double p, double q, double _d_y, double* _d_p,
                   double* _d_q);
} // namespace custom_derivatives
} // namespace clad

#ifdef CROSSTU_DEFINITIONS

double kick(double p, double q) { return p * p * q; }

namespace clad {
namespace custom_derivatives {
void kick_pullback(double p, double q, double _d_y, double* _d_p,
                   double* _d_q) {
  *_d_p += 2 * p * q * _d_y;
  *_d_q += p * p * _d_y;
}
} // namespace custom_derivatives
} // namespace clad

#else

double f(double x, double y) { return kick(x, y); }

// CHECK: void f_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: clad::custom_derivatives::kick_pullback(x, y, 1, &_r0, &_r1);

int main() {
  double dx = 0, dy = 0;
  clad::gradient(f).execute(3, 4, &dx, &dy);
  printf("%.2f %.2f\n", dx, dy); // CHECK-EXEC: 24.00 9.00
}

#endif
