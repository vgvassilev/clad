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
}
