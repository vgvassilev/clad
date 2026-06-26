// RUN: %cladclang -fsyntax-only -std=c++17 -I%S/../../include %s

#include "clad/Differentiator/Differentiator.h"

int* global_ptr;

void use(int*) {}

double fn(double x) {
  use(global_ptr);
  return x * x;
}

void test() {
  auto grad_fn = clad::gradient(fn);
}
