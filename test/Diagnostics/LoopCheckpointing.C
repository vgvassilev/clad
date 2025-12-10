// RUN: %cladclang %s -I%S/../../include -Xclang -verify -c

#include "clad/Differentiator/Differentiator.h"

double fn_dangling_checkpoint(double x, double y) {
  #pragma clad checkpoint loop  // expected-error {{'#pragma clad checkpoint loop' is only allowed before a loop}}
  return x + 1;
}

int main() {
    clad::gradient(fn_dangling_checkpoint);
}