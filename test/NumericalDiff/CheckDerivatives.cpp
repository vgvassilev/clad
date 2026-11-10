// Checking what clad generated against finite differences of the primal.
// RUN: %cladnumdiffclang %s -I%S/../../include -oCheckDerivatives.out -Xclang -verify 2>&1
// RUN: ./CheckDerivatives.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

extern "C" int printf(const char* fmt, ...);

// A smooth function: the two derivatives agree everywhere.
double smooth(double x, double y) { return x * x * y + std::sin(y); }

// The same thing with a term added. Checking smooth's gradient against this
// one stands in for a derivative that is wrong: the check has to catch it,
// which is the whole point of having it.
double shifted(double x, double y) { return x * x * y + std::sin(y) + x; }

// Not smooth at x = 0, where the finite difference of |x| is 0 and the
// derivative is not. The report says they disagree; which of the two is wrong
// is the caller's to judge, and here it is the finite difference.
double kink(double x, double y) { return std::abs(x) + y; }

// An array input: one derivative per element, and the length comes from the
// array_ref the caller wraps it in.
double over_array(const double* a, double y) {
  return a[0] * a[1] + a[2] * y;
}

// An integer input takes no small step, so it is carried in the report as
// unchecked rather than counted as a failure.
double counted(double x, int n) { return x * n; }

int main() { // expected-no-diagnostics
  auto dSmooth = clad::gradient(smooth);
  clad::check_report agree = clad::check_gradient(smooth, dSmooth, 3.0, 2.0);
  printf("smooth: ok=%d entries=%d\n", (int)agree.ok(),
         (int)agree.entries().size());
  // CHECK-EXEC: smooth: ok=1 entries=2
  agree.print();
  // CHECK-EXEC: clad: 2 of 2 derivatives agree with finite differences

  clad::check_report caught = clad::check_gradient(shifted, dSmooth, 3.0, 2.0);
  printf("wrong: ok=%d mismatches=%d\n", (int)caught.ok(),
         (int)caught.mismatches());
  // CHECK-EXEC: wrong: ok=0 mismatches=1
  caught.print();
  // CHECK-EXEC: clad: 1 of 2 derivatives disagree with finite differences (abs tol 1e-06, rel tol 1e-05)

  // The tolerances are the caller's to set, and a loose enough pair accepts
  // the same mismatch.
  clad::check_options loose;
  loose.abs_tolerance = 2;
  loose.rel_tolerance = 0;
  clad::check_report accepted =
      clad::check_gradient(loose, shifted, dSmooth, 3.0, 2.0);
  printf("loose: ok=%d\n", (int)accepted.ok());
  // CHECK-EXEC: loose: ok=1

  auto dKink = clad::gradient(kink);
  clad::check_report atKink = clad::check_gradient(kink, dKink, 0.0, 1.0);
  printf("kink: ok=%d mismatches=%d\n", (int)atKink.ok(),
         (int)atKink.mismatches());
  // CHECK-EXEC: kink: ok=0 mismatches=1
  // Away from the kink the same function agrees.
  clad::check_report offKink = clad::check_gradient(kink, dKink, 2.0, 1.0);
  printf("off kink: ok=%d\n", (int)offKink.ok());
  // CHECK-EXEC: off kink: ok=1

  double a[3] = {1.5, 2.5, 3.5};
  auto dArray = clad::gradient(over_array);
  clad::check_report array = clad::check_gradient(
      over_array, dArray, clad::array_ref<double>(a, 3), 2.0);
  printf("array: ok=%d entries=%d\n", (int)array.ok(),
         (int)array.entries().size());
  // CHECK-EXEC: array: ok=1 entries=4

  auto dCounted = clad::gradient(counted);
  clad::check_report counts = clad::check_gradient(counted, dCounted, 3.0, 4);
  printf("integer input: ok=%d unchecked=%d\n", (int)counts.ok(),
         (int)counts.unchecked());
  // CHECK-EXEC: integer input: ok=1 unchecked=1
  counts.print();
  // CHECK-EXEC: clad: 1 of 2 derivatives agree with finite differences, 1 not checked
  // CHECK-EXEC: arg 1: not checked, an integer or enum input takes no step

  // Forward mode: the first input by default, another one on request.
  auto dx = clad::differentiate(smooth, "x");
  clad::check_report forward = clad::check_derivative(smooth, dx, 3.0, 2.0);
  printf("forward: ok=%d entries=%d\n", (int)forward.ok(),
         (int)forward.entries().size());
  // CHECK-EXEC: forward: ok=1 entries=1

  auto dy = clad::differentiate(smooth, "y");
  clad::check_options wrtY;
  wrtY.wrt = 1;
  clad::check_report forwardY =
      clad::check_derivative(wrtY, smooth, dy, 3.0, 2.0);
  printf("forward wrt y: ok=%d\n", (int)forwardY.ok());
  // CHECK-EXEC: forward wrt y: ok=1

  // A derivative of y checked as though it were of x disagrees, which is the
  // mistake the wrt option exists to let the caller avoid.
  clad::check_report mismatched = clad::check_derivative(smooth, dy, 3.0, 2.0);
  printf("forward wrt wrong input: ok=%d\n", (int)mismatched.ok());
  // CHECK-EXEC: forward wrt wrong input: ok=0
}
