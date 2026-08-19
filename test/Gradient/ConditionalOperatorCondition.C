// RUN: %cladclang %s -I%S/../../include -oConditionalOperatorCondition.out 2>&1 | %filecheck %s
// RUN: ./ConditionalOperatorCondition.out | %filecheck_exec %s
//
// A conditional operator's condition is a boolean context, so clad must store
// it as bool. Storing the raw operand dropped qualifiers: a `const char*`
// condition became an ill-formed `char* _cond = <const char*>` -- exactly what
// libstdc++'s basic_string(const char*) constructor triggers under gcc-11.
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double f(double x, const char *s) { return s ? x * x : x; }

// CHECK: bool _cond0 = s;

int main() {
  auto g = clad::gradient(f, "x");
  double dx = 0;
  g.execute(3.0, "x", &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
}
