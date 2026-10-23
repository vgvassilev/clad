// RUN: %cladclang %s -I%S/../../include -oBaseInitList.out
// RUN: ./BaseInitList.out | %filecheck_exec %s

// The initializer list of an aggregate with a base class initializes the
// base first, then the fields.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct Base { double v; };
struct Derived : Base { int n; };

// f = 2 k^2, f' = 4 k.
double f(double k) {
  Derived a{{k}, 2};
  return a.v * a.n * k;
}

int main() {
  double dk = 0;
  clad::gradient(f).execute(3, &dk);
  printf("%.2f\n", dk); // CHECK-EXEC: 12.00
}
