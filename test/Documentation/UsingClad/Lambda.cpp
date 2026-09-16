// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-lambda
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

int main() {
  auto lambda = [](double i, double j) { return i * j; };
  // Pass by reference.
  auto lambda_grad = clad::gradient(lambda);
  // Can be passed by pointer as well!
  auto lambda_grad_pointer = clad::gradient(&lambda);

  double d_i_1, d_j_1, d_i_2, d_j_2;
  d_i_1 = d_j_1 = d_i_2 = d_j_2 = 0;

  lambda_grad.execute(3, 5, &d_i_1, &d_j_1);
  lambda_grad_pointer.execute(3, 5, &d_i_2, &d_j_2);

  std::cout << d_i_1 << " " << d_j_1 << "\n"; // prints: 5 3
  std::cout << d_i_2 << " " << d_j_2 << "\n"; // prints: 5 3
}
// docs-end-lambda
