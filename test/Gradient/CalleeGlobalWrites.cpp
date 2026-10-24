// RUN: %cladclang %s -I%S/../../include -oCalleeGlobalWrites.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./CalleeGlobalWrites.out | %filecheck_exec %s

// The pullback of a call replays the globals the call assigns. The reverse
// sweep puts them back around it, as the call found them, and the gradient
// leaves them as the function itself would.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// CHECK: warning: gradient uses a global variable 'counter'; rerunning the gradient requires 'counter' to be reset
int counter = 1;
// CHECK-DAG: warning: gradient uses a global variable 'first'; rerunning the gradient requires 'first' to be reset
int first = 1;
// CHECK-DAG: warning: gradient uses a global variable 'second'; rerunning the gradient requires 'second' to be reset
int second = 1;

void step(double& x) {
  counter += 1;
  x *= counter;
}

double f(double x) {
  for (int i = 0; i < 2; ++i)
    step(x);
  return x;
}

// CHECK: void f_grad(double x, double *_d_x) {
// CHECK: _t2[i] = counter;
// CHECK-NEXT: step(x);
// CHECK: int _final0 = counter;
// CHECK: counter = _t2[i];
// CHECK-NEXT: step_pullback(x, _d_x);
// CHECK-NEXT: counter = _t2[i];
// CHECK-NEXT: }
// CHECK-NEXT: counter = _final0;

// With an early return the forward sweep runs in a closure. The copy of the
// global is taken there, when the call is reached, not ahead of the closure.
double early(double x, double c) {
  if (c < 0)
    return 0;
  step(x);
  return x;
}

// CHECK: void early_grad_0(double x, double c, double *_d_x) {
// CHECK: int _t1 = 0;
// CHECK: clad::forward_sweep([&] {
// CHECK: _t1 = counter;
// CHECK-NEXT: step(x);

// The globals are put back in the order they are declared.
void twoGlobals(double& x) {
  second += 1;
  first += 1;
  x *= first * second;
}

double g(double x) {
  twoGlobals(x);
  return x;
}

// CHECK: void g_grad(double x, double *_d_x) {
// CHECK: int _t{{[0-9]+}} = first;
// CHECK-NEXT: int _t{{[0-9]+}} = second;
// CHECK-NEXT: twoGlobals(x);

int main() {
  double dx = 0;
  // f = 2 * 3 * x
  clad::gradient(f).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  printf("%d\n", counter); // CHECK-EXEC: 3
  dx = 0;
  // counter is 3 now, so f = 4 x
  clad::gradient(early, "x").execute(3, 1, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 4.00
  dx = 0;
  clad::gradient(g).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 4.00
}
