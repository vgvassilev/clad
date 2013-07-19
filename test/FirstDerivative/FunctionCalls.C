// RUN: %autodiff %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s
//XFAIL:*
#include "autodiff/Differentiator/Differentiator.h"

int printf(const char* fmt, ...);
int custom_fn(int x);

namespace custom_derivatives {
  int custom_fn(int x) {
    return x + x;
  }
}

int a(int x) {
  printf("A was called.\n");
  return x*x;
}

float b(int x) {
  return a(x) + custom_fn(x);
}

// CHECK: float  b_derived_x(int x) {
// CHECK-NEXT: return a_derived(x) + (custom_fn(x));
// CHECK-NEXT: }

int main () {
  int x = 4;
  diff(b, 1);

  return 0;
}
