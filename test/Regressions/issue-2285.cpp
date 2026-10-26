// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t
// RUN: %t | %filecheck %s --check-prefix=CHECK-EXEC
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double rangeBreak(double x) {
  double s = 1;
  double a[4] = {1, 2, 3, 4};
  for (double v : a) {
    if (v == 3)
      break;
    s *= x * v;
  }
  return s;
}

double rangeActiveValues(double x) {
  double s = 1;
  double a[4] = {x, 2 * x, 3 * x, 4 * x};
  for (double v : a) {
    if (v == 3 * x)
      break;
    s *= x * v;
  }
  return s;
}

double rangeReturn(double x) {
  double s = 1;
  double a[4] = {1, 2, 3, 4};
  for (double v : a) {
    if (v == 7)
      return s * x;
    if (v == 3)
      return s;
    s *= x * v;
  }
  return s;
}

double rangeControl(double x, int stop, int skip) {
  double s = 1;
  double a[4] = {1, 2, 3, 4};
  for (double v : a) {
    if (v == stop)
      break;
    if (v == skip)
      continue;
    s *= x * v;
  }
  return s;
}

int main() {
  double dx = 0;
  clad::gradient(rangeBreak).execute(1.3, &dx);
  printf("range %.2f\n", dx);
  // CHECK-EXEC: range 5.20
  dx = 0;
  clad::gradient(rangeActiveValues).execute(1.3, &dx);
  printf("active range %.2f\n", dx);
  // CHECK-EXEC-NEXT: active range 17.58
  dx = 0;
  clad::gradient(rangeReturn).execute(1.3, &dx);
  printf("range return %.2f\n", dx);
  // CHECK-EXEC-NEXT: range return 5.20
  auto dControl = clad::gradient(rangeControl, "x");
  dx = 0;
  dControl.execute(1.3, 1, 0, &dx);
  printf("first break %.2f\n", dx);
  // CHECK-EXEC-NEXT: first break 0.00
  dx = 0;
  dControl.execute(1.3, 0, 2, &dx);
  printf("continue %.2f\n", dx);
  // CHECK-EXEC-NEXT: continue 60.84
  dx = 0;
  dControl.execute(1.3, 0, 0, &dx);
  printf("no break %.2f\n", dx);
  // CHECK-EXEC-NEXT: no break 210.91
}
