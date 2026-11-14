// RUN: %cladclang %s -I%S/../../include -oRValueReferences.out
// RUN: ./RValueReferences.out | %filecheck_exec %s

// A reference bound to an xvalue refers to the object, so its adjoint is that
// object's adjoint.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#include <utility>

struct D {
  double v;
};

double moved(double x) {
  D d;
  d.v = x * x;
  D&& r = std::move(d);
  return r.v;
}

double constMoved(double x) {
  double t = x * x;
  const double& r = std::move(t);
  return r;
}

double inBlock(double x) {
  D d;
  d.v = x * x;
  double s = 0;
  if (x > 0) {
    D&& r = std::move(d);
    s = r.v;
  }
  return s;
}

double inLoop(double x) {
  double s = 0;
  for (int i = 0; i < 3; ++i) {
    double t = x * i;
    double&& r = static_cast<double&&>(t);
    s += r * r;
  }
  return s;
}

// A call returning an rvalue reference returns one to the adjoint too.
double&& first(D&& d) { return static_cast<double&&>(d.v); }
double fromCall(double x) {
  D d;
  d.v = x;
  double&& r = first(std::move(d));
  return r * x;
}

// So does one returning an rvalue reference to const: the reference adjoint
// it returns refers into its adjoint parameter, which it writes through.
const double&& firstOfConst(const D&& d) {
  return static_cast<const double&&>(d.v);
}
double fromConstCall(double x) {
  const D d{x};
  const double&& r = firstOfConst(std::move(d));
  return r * x;
}

int main() {
  double dx = 0;
  clad::gradient(moved).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(constMoved).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(inBlock).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(inLoop).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 30.00
  dx = 0;
  clad::gradient(fromCall).execute(3, &dx);
  printf("%.2f %.2f\n", dx, clad::differentiate(fromCall, 0).execute(3));
  // CHECK-EXEC: 6.00 6.00
  dx = 0;
  clad::gradient(fromConstCall).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
}
