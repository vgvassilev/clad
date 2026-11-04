// RUN: %cladclang %s -I%S/../../include -oissue-2082.out
// RUN: ./issue-2082.out | %filecheck_exec %s

// The adjoint of a reference member refers to the adjoint of what the member
// refers to, so binding it passes no adjoint back. A reference to const is
// written through all the same, and bound to a temporary it refers to the
// adjoint of that temporary.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct Ref {
  double &a, &b;
  Ref(double& x, double& y) : a(x), b(y) {}
};

struct Agg {
  double &a, &b;
};

struct ConstRef {
  const double& a;
  double b;
};

double temporary(double x) {
  ConstRef r{x * x, x};
  return r.a * r.b;
}

double ctor(double x, double y) {
  Ref r(x, y);
  return r.a * r.b;
}

double aggregate(double x, double y) {
  Agg r{x, y};
  return r.a * r.b;
}

int main() {
  double dx = 0, dy = 0;
  clad::gradient(ctor).execute(3, 4, &dx, &dy);
  printf("%.2f %.2f\n", dx, dy); // CHECK-EXEC: 4.00 3.00
  dx = dy = 0;
  clad::gradient(aggregate).execute(3, 4, &dx, &dy);
  printf("%.2f %.2f\n", dx, dy); // CHECK-EXEC: 4.00 3.00
  dx = 0;
  clad::gradient(temporary).execute(2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 12.00
}
