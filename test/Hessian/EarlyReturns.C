// RUN: %cladclang %s -I%S/../../include -oEarlyReturns.out 2>&1 | %filecheck %s
// RUN: ./EarlyReturns.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oEarlyReturns.out
// RUN: ./EarlyReturns.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

double mat3[9];
clad::array_ref<double> mat3_ref(mat3, 9);
double mat4[16];
clad::array_ref<double> mat4_ref(mat4, 16);

// For degree == 2 the early return is taken and the loop is never entered,
// so the forward sweep neither pushes onto the loop's tapes nor resets its
// counter. The master reverse sweep still runs on that path, so the counter
// has to read zero there -- otherwise the reverse loop iterates a garbage
// number of times and pops empty tapes (vgvassilev/clad#1985).
inline double poly(double* coefs, int n) {
  int degree = n - 1;

  if (degree == 2)
    return coefs[2] * coefs[1] + coefs[1] * coefs[0];

  double t = 1.;
  double result = coefs[0];
  for (int i = 1; i < degree; i++) {
    result = result + t * coefs[i];
    t *= coefs[0];
  }
  return result;
}

// Takes the early return: c2 * c1 + c1 * c0.
double earlyBeforeLoop(double* c) { return poly(c, 3); }

// Falls through to the loop: c0 + c1 + c0 * c2.
double loopAfterEarly(double* c) { return poly(c, 4); }

// CHECK-LABEL: inline void poly_pushforward_pullback(
// The loop counter is hoisted above the master reverse lambda and only reset
// in the forward sweep, so it must be zero-initialized at its declaration.
// CHECK: {{__size_t|size_t|unsigned long long|unsigned long|unsigned int}} _t0 = {{0U|0UL|0ULL}};
// CHECK: auto _rev0 = [&
// CHECK: for (; _t0; _t0--)

int main() {
  double c[4] = {0.3, 0.7, 0.2, 0.5};

  INIT_HESSIAN(earlyBeforeLoop, "c[0:2]");
  INIT_HESSIAN(loopAfterEarly, "c[0:3]");

  // d(c2 * c1 + c1 * c0) = {c1, c2 + c0, c1}.
  TEST_HESSIAN(earlyBeforeLoop, 1, c, mat3_ref);
  // CHECK-EXEC: {0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00}

  // d(c0 + c1 + c0 * c2) = {1 + c2, 1, c0, 0}.
  TEST_HESSIAN(loopAfterEarly, 1, c, mat4_ref);
  // CHECK-EXEC: {0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}
}
