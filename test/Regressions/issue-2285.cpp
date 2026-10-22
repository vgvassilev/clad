// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t
// RUN: %t | %filecheck %s --check-prefix=CHECK-EXEC
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double nestedExit(double x) {
  double s = 1;
  for (int i = 0; i < 4; i++) {
    s *= x;
    if (i > 0) {
      if (i == 2)
        return s;
    }
    s *= x;
  }
  return s;
}

double chainedExit(double x) {
  double s = 1;
  for (int i = 0; i < 5; i++) {
    s *= x;
    if (i == 7)
      return s;
    else if (i == 2)
      return s * x;
    s *= x;
  }
  return s;
}

double effectfulExit(double x) {
  double s = 1;
  for (int i = 0; i < 3; i++) {
    if (i == 1)
      return s *= x;
    s *= x;
  }
  return s;
}

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

double switchExit(double x) {
  switch (int(x)) {
  case 1:
    return x * x;
  default:
    return x;
  }
}
void outputExit(double x, double* out) {
  for (int i = 0; i < 4; i++) {
    if (i == 1)
      continue;
    out[0] *= x;
    if (i == 2)
      return;
    out[0] += x;
  }
}

int main() {
  double dx = 0;
  clad::gradient(nestedExit).execute(1.3, &dx);
  printf("nested %.2f\n", dx);
  // CHECK-EXEC: nested 14.28
  dx = 0;
  clad::gradient(chainedExit).execute(1.3, &dx);
  printf("chained %.2f\n", dx);
  // CHECK-EXEC-NEXT: chained 22.28
  dx = 0;
  clad::gradient(effectfulExit).execute(1.3, &dx);
  printf("effectful %.2f\n", dx);
  // CHECK-EXEC-NEXT: effectful 2.60
  dx = 0;
  clad::gradient(rangeBreak).execute(1.3, &dx);
  printf("range %.2f\n", dx);
  // CHECK-EXEC-NEXT: range 5.20
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
  dx = 0;
  clad::gradient(switchExit).execute(1.3, &dx);
  printf("switch %.2f\n", dx);
  // CHECK-EXEC-NEXT: switch 2.60
  dx = 0;
  double out[1] = {2}, dout[1] = {1};
  clad::gradient(outputExit).execute(1.3, out, &dx, dout);
  printf("output %.2f\n", dx);
  // CHECK-EXEC-NEXT: output 7.80
}
