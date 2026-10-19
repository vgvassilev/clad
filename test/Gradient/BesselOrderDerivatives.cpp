// RUN: %cladclang -std=c++17 %s -I%S/../../include -oBesselOrderDerivatives.out
// RUN: ./BesselOrderDerivatives.out | %filecheck_exec %s
// RUN: %cladclang -std=c++17 -UCLAD_NO_NUM_DIFF -DNUM_DIFF %s \
// RUN:   -I%S/../../include -oBesselOrderDerivativesNum.out
// RUN: ./BesselOrderDerivativesNum.out | %filecheck_exec %s
// CHECK-EXEC: Bessel order derivatives: {{PASS|SKIP}}

// The derivative of cyl_bessel_j, _i and _k with respect to the order nu, in
// forward and reverse mode, against mpmath's derivative in nu at x = 2.5.
// Integer orders use closed forms. A non-integer order needs a numerical
// difference, which CLAD_NO_NUM_DIFF turns into NaN instead of zero.

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cstdio>

#ifdef __cpp_lib_math_special_functions
double j(double nu, double x) { return std::cyl_bessel_j(nu, x); }
double i(double nu, double x) { return std::cyl_bessel_i(nu, x); }
double k(double nu, double x) { return std::cyl_bessel_k(nu, x); }
double pair(double x, double y) {
  return 2.0 * std::cyl_bessel_j(0.0, x) - 3.0 * std::cyl_bessel_j(0.5, y);
}

static int failures = 0;

static void expect(const char* what, double nu, double got, double want,
                   double tol) {
  bool ok = std::isnan(want) ? std::isnan(got)
                             : std::fabs(got - want) <= tol * (1 + std::fabs(want));
  if (!ok) {
    printf("FAIL %s nu=%g: got %.12g, want %.12g\n", what, nu, got, want);
    ++failures;
  }
}

#define ORDER(F, NU, WANT, TOL)                                                \
  {                                                                            \
    double dnu = 0, dx = 0;                                                    \
    clad::gradient(F).execute(NU, 2.5, &dnu, &dx);                             \
    expect(#F " reverse", NU, dnu, WANT, TOL);                                 \
    expect(#F " forward", NU, clad::differentiate(F, "nu").execute(NU, 2.5),   \
           WANT, TOL);                                                         \
  }

int main() {
  ORDER(j, 0.0, 0.782367091369, 1e-11);
  ORDER(j, 1.0, 0.209854164544, 1e-11);
  ORDER(j, 3.0, -0.199900482212, 1e-11);
  ORDER(i, 0.0, -0.0623475532004, 1e-11);
  ORDER(i, 1.0, -1.24204484127, 1e-11);
  ORDER(i, 3.0, -0.531882277266, 1e-11);
  ORDER(k, 0.0, 0.0, 1e-11);
  ORDER(k, 1.0, 0.0249390212801, 1e-11);
  ORDER(k, 3.0, 0.248609378467, 1e-11);
#ifdef NUM_DIFF
  ORDER(j, 0.7, 0.427230992235, 1e-7);
  ORDER(i, 0.7, -1.04715407433, 1e-7);
  ORDER(k, 0.7, 0.0161198801311, 1e-7);
#else
  ORDER(j, 0.7, NAN, 0);
  ORDER(k, 0.7, NAN, 0);
#endif
  // An order no input varies must not reach the other directions, even when
  // its own derivative would be NaN: vector mode hands it a zero array.
  double dx = 0, dy = 0;
  clad::differentiate<clad::opts::vector_mode>(pair).execute(1.0, 2.0, &dx,
                                                            &dy);
  expect("vector d/dx", 0.5, dx, -2 * std::cyl_bessel_j(1.0, 1.0), 1e-12);
  expect("vector d/dy", 0.5, dy,
         -3 * (-std::cyl_bessel_j(1.5, 2.0) + 0.25 * std::cyl_bessel_j(0.5, 2.0)),
         1e-12);
  if (!failures)
    printf("Bessel order derivatives: PASS\n");
}
#else
int main() { printf("Bessel order derivatives: SKIP\n"); }
#endif
