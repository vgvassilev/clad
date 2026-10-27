// RUN: %cladclang %s -I%S/../../include -oPushforwardCallees.out
// RUN: ./PushforwardCallees.out | %filecheck_exec %s

// Reverse mode differentiates a real function of one argument through its
// pushforward, whose callees then need a pushforward too.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double two(double a, double b) { return a * b; }

double through_two(double x) {
  return two(1.0, x);
}
double f1(double x) { return through_two(x); }

double through_lambda(double x) {
  auto mul = [](double a, double b) { return a * b; };
  return mul(1.0, x);
}
double f2(double x) { return through_lambda(x); }

int main() {
  double d = 0;
  clad::gradient(f1).execute(1.5, &d);
  printf("%.2f\n", d); // CHECK-EXEC: 1.00
  d = 0;
  clad::gradient(f2).execute(1.5, &d);
  printf("%.2f\n", d); // CHECK-EXEC: 1.00
}
