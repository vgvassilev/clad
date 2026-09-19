// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-member-overloads
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

class A {
public:
  float f1(float x) { return x + x + x; }
  double f1(double x) { return x + x + x + x; }
};

int main() {
  auto d_f1 = clad::differentiate(static_cast<float (A::*)(float)>(&A::f1), 0);

  A a;
  std::cout << d_f1.execute(a, 2.0f) << "\n"; // prints: 3
}
// docs-end-member-overloads
