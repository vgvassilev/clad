// RUN: %cladclang %s -I%S/../../include -oLocalPointerAlias.out
// RUN: ./LocalPointerAlias.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oLocalPointerAlias.out
// RUN: ./LocalPointerAlias.out | %filecheck_exec %s

// A callee that stores through a pointer kept in a local array or struct
// writes the caller's buffer, and its pullback needs the value from before the
// call: the store has to be recorded, not taken for one into a local.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

void viaArray(double* out) {
  double* ptrs[1] = {out};
  ptrs[0][1] = ptrs[0][1] * ptrs[0][1];
}

struct Holder {
  double* p;
};
void viaMember(double* out) {
  Holder h{out};
  h.p[1] = h.p[1] * h.p[1];
}

// An increment through such a pointer writes the caller's buffer as well.
void bump(double* out) {
  double* ptrs[1] = {out};
  ++ptrs[0][1];
}

// f = (x^2)^2 = x^4, f' = 4x^3.
double array(double x) {
  double o[2] = {0, x};
  viaArray(o);
  viaArray(o);
  return o[1];
}

double member(double x) {
  double o[2] = {0, x};
  viaMember(o);
  viaMember(o);
  return o[1];
}

// f = (x + 1)^2, f' = 2(x + 1).
double increment(double x) {
  double o[2] = {0, x};
  bump(o);
  viaArray(o);
  return o[1];
}

int main() {
  double d = 0;
  clad::gradient(array).execute(1.5, &d);
  printf("array %.4f\n", d); // CHECK-EXEC: array 13.5000
  d = 0;
  clad::gradient(member).execute(1.5, &d);
  printf("member %.4f\n", d); // CHECK-EXEC: member 13.5000
  d = 0;
  clad::gradient(increment).execute(1.5, &d);
  printf("increment %.4f\n", d); // CHECK-EXEC: increment 5.0000
}
