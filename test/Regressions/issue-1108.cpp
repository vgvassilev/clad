// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s

#include <cstdio>
#include <cmath>
#include "clad/Differentiator/Differentiator.h"

/**
 * @brief The function to be differentiated.
 * f(C, A) = 2 * C^2 * A
 */
double f(double C, double A) {
  double a = std::acos(-C / A);
  (void)a; 
  return 2 * C * C * A;
}

int main() {
  double C_val = 1.0;
  double A_val = 2.0;
  double dC = 0, dA = 0;

  // 1. Test clad::gradient
  auto f_grad = clad::gradient(f);
  f_grad.execute(C_val, A_val, &dC, &dA);

  printf("Gradient: %.2f %.2f\n", dC, dA);
  // CHECK-EXEC: Gradient: 8.00 2.00

  // 2. Test clad::differentiate
  auto f_dC = clad::differentiate(f, "C");
  auto f_dA = clad::differentiate(f, "A");
  
  printf("Differentiate: %.2f %.2f\n", 
         f_dC.execute(C_val, A_val), 
         f_dA.execute(C_val, A_val));
  // CHECK-EXEC: Differentiate: 8.00 2.00

  return 0;
}