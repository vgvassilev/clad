// RUN: %cladclang %s -I%S/../../include -oNonDifferentiableMarkedType.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./NonDifferentiableMarkedType.out | %filecheck_exec %s
//
// A user type marked non-differentiable with CLAD_NONDIFFERENTIABLE_TYPE is
// opaque: callOperatesOnNonDifferentiableType makes clad treat a call that
// operates on it as carrying no derivative instead of descending into it. The
// "type the call operates on" is the object of a member call (Meter::read
// below) or the first argument of a free operator (operator<< on Logger).

#include "clad/Differentiator/Differentiator.h"

struct Meter {
  double v;
  double read() const { return v; }
};
CLAD_NONDIFFERENTIABLE_TYPE(Meter);

double f(double x) {
  Meter m{5.0};
  return x * x + m.read(); // m.read() is opaque -> zero derivative contribution
}

// CHECK: void f_grad(double x, double *_d_x) {
// CHECK-NOT: read
// CHECK: *_d_x += 1 * x;
// CHECK-NEXT: *_d_x += x * 1;

struct Logger {};
CLAD_NONDIFFERENTIABLE_TYPE(Logger);
// Free operator whose first argument is the marked type -- the arg0 branch of
// callOperatesOnNonDifferentiableType, distinct from the member-call branch
// above.
Logger& operator<<(Logger& l, double) { return l; }

double g(double x) {
  Logger log;
  log << (x * x); // opaque -> emitted as-is, feeds nothing into the adjoint
  return x * x * x;
}

// The operator call survives verbatim in the derivative and gets no pullback;
// only the return contributes to the adjoint (d/dx x^3 = 3x^2 = 12 at x = 2).
// CHECK: void g_grad(double x, double *_d_x) {
// CHECK: {{.*}} << (x * x);
// CHECK-NOT: pullback

int main() {
  auto df = clad::gradient(f);
  double dx = 0;
  df.execute(3.0, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00

  auto dg = clad::gradient(g);
  dx = 0;
  dg.execute(2.0, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 12.00
}

// expected-no-diagnostics
