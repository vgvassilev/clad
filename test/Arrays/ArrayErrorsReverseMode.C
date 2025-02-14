// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1

#include "clad/Differentiator/Differentiator.h"

float func11(float* a, float b) { // expected-error {{Non-differentiable non-const pointer and array parameters are not supported. Please differentiate w.r.t. 'a' or mark it const.}}
  float sum = 0;
  sum += a[0] *= b;
  return sum;
}

float func12(float a, float b[]) { // expected-error {{Non-differentiable non-const pointer and array parameters are not supported. Please differentiate w.r.t. 'b' or mark it const.}}
  float sum = 0;
  sum += a *= b[1];
  return sum;
}

int main() {
  clad::gradient(func11, "b");
  clad::gradient(func12, "a");
}