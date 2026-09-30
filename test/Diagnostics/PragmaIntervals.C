// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

double square(double x) { return x * x; }

int main() {
#pragma clad OFF
  // expected-warning@+1 {{clad differentiation call is disabled by '#pragma clad OFF'}}
  clad::gradient(square);
#pragma clad ON
  square(3);
}
