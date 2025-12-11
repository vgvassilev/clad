// RUN: %clangxx -fplugin=%clad_plugin -std=c++17 %s -o %t
// RUN: %t | FileCheck %s

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <iostream>
#include <numeric>

// Wrapper function to differentiate
double ellint_k(double k) {
  return std::comp_ellint_1(k);
}

int main() {
  double k = 0.5;

  // --- Test 1: Forward Mode (differentiate) ---
  auto ellint_diff = clad::differentiate(ellint_k, "k");
  double deriv_forward = ellint_diff.execute(k);

  std::cout << "Forward Derivative: " << deriv_forward << std::endl;
  // CHECK: Forward Derivative: 0.541732

  // --- Test 2: Reverse Mode (gradient) ---
  auto ellint_grad = clad::gradient(ellint_k, "k");
  double d_k = 0;
  ellint_grad.execute(k, &d_k);

  std::cout << "Reverse Derivative: " << d_k << std::endl;
  // CHECK: Reverse Derivative: 0.541732

  return 0;
}
