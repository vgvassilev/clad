// RUN: %cladclang %s -I%S/../../include -Xclang -verify -c

#include "clad/Differentiator/Differentiator.h"

double fn_dangling_checkpoint(double x, double y) {
  #pragma clad checkpoint loop  // expected-error {{'#pragma clad checkpoint loop' is only allowed before a loop}}
  return x + 1;
}

double fn_mixed_checkpoint(double x, double y) {
  #pragma clad checkpoint loop
  for (int i = 0; i < 2; ++i)
    x += y;

  #pragma clad checkpoint loop  // expected-error {{'#pragma clad checkpoint loop' is only allowed before a loop}}
  return x;
}

int main() {
  // Hits duplicate pragma-diagnosis suppression path.
  clad::gradient(fn_dangling_checkpoint);
  clad::hessian(fn_dangling_checkpoint, "x");

  // Hits reverse loop checkpoint scan with one invalid entry in map.
  clad::gradient(fn_mixed_checkpoint);
}
