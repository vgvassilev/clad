// RUN: %cladclang -std=c++20 -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s
// UNSUPPORTED: clang-10, clang-11, clang-12, clang-13, clang-14, clang-15, clang-16
// XFAIL: valgrind

#include <iostream>
#include "clad/Differentiator/Differentiator.h"

#pragma clad ON
double f(double* params, const double* obs) {
  double res = 0.0;
  const double t1 = params[0] + params[1];
  #pragma clad checkpoint loop
  for (int i = 0; i < (int)obs[0]; ++i) {
    res += t1 * obs[i + 1];
  }
  return res;
}

void request() {
  clad::gradient(f, "params");
  clad::hessian(f, "params[0:1]");
}

#pragma clad OFF

int main() {
  request();
  std::cout << "ok\n";
  // CHECK-EXEC: ok
  return 0;
}
