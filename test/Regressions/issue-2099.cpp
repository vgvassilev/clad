// RUN: %cladclang %s -I%S/../../include -oissue-2099.out
// RUN: ./issue-2099.out | %filecheck_exec %s

// The adjoint of a cast to another class has that class's type.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct B {
  double v;
  virtual ~B() = default;
};
struct D : B {};

double down(double x) {
  D d;
  d.v = x * x;
  B* b = &d;
  D* p = static_cast<D*>(b);
  return p->v;
}

double dynamicDown(double x) {
  D d;
  d.v = x * x;
  B* b = &d;
  D* p = dynamic_cast<D*>(b);
  return p->v;
}

double refDown(double x) {
  D d;
  d.v = x * x;
  B& b = d;
  D& r = static_cast<D&>(b);
  return r.v;
}

int main() {
  double dx = 0;
  clad::gradient(down).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(dynamicDown).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(refDown).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
}
