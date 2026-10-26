// RUN: %cladclang %s -I%S/../../include -oByteCopyCustomAssign.out
// RUN: ./ByteCopyCustomAssign.out | %filecheck_exec %s

// The default assignment of a class with an array member copies it with
// memcpy, which clad refuses. Custom derivatives for the assignment replace
// that body, so it is not an error.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct M { double m[2]; };

namespace clad::custom_derivatives::class_functions {
template <class A>
clad::ValueAndAdjoint<M&, M&> operator_equal_reverse_forw(M* self, A&& other,
                                                         M* d_self, A&&,
                                                         restore_tracker& t) {
  t.store(*self);
  *self = other;
  return {*self, *d_self};
}
template <class A>
void operator_equal_pullback(M* self, A&& other, M* d_self, M* d_other) {
  for (int i = 0; i < 2; ++i) {
    d_other->m[i] += d_self->m[i];
    d_self->m[i] = 0;
  }
}
} // namespace clad::custom_derivatives::class_functions

M twice(const M& a) { return {{2 * a.m[0], 2 * a.m[1]}}; }
void push(M& a) { a = twice(a); }

// f = 2 k^2, f' = 4 k.
double f(double k) {
  M a{{k, k}};
  push(a);
  return a.m[0] * k;
}

int main() {
  double dk = 0;
  clad::gradient(f).execute(3, &dk);
  printf("%.2f\n", dk); // CHECK-EXEC: 12.00
}
