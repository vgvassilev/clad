// RUN: %cladclang -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double sum(double u, double v) {
  clad::array<double> p = {u, v};
  return p[0] + p[1];
}

double product(double u, double v) {
  clad::array<double> p = {u, v, u};
  return p[0] * p[1] * p[2];
}

double loop(double u, double v) {
  double s = 0;
  for (int i = 0; i < 3; ++i) {
    clad::array<double> p = {u * i, v};
    s += p[0] * p[1];
  }
  return s;
}

float single(float u, float v) {
  clad::array<float> p = {u, v, u};
  return p[0] * p[1] + p[2];
}

double constant(double u, double v) {
  clad::array<double> c = {1.0, 2.0};
  return c[0] * u + c[1] * v;
}

int main() {
  double du = 0, dv = 0;
  clad::gradient(sum).execute(3, 5, &du, &dv);
  printf("%.2f %.2f\n", du, dv); // CHECK-EXEC: 1.00 1.00

  du = dv = 0;
  clad::gradient(product).execute(3, 5, &du, &dv);
  printf("%.2f %.2f\n", du, dv); // CHECK-EXEC: 30.00 9.00

  du = dv = 0;
  clad::gradient(loop).execute(3, 5, &du, &dv);
  printf("%.2f %.2f\n", du, dv); // CHECK-EXEC: 15.00 9.00

  float fu = 0, fv = 0;
  clad::gradient(single).execute(3, 5, &fu, &fv);
  printf("%.2f %.2f\n", fu, fv); // CHECK-EXEC: 6.00 3.00

  du = dv = 0;
  clad::gradient(constant).execute(3, 5, &du, &dv);
  printf("%.2f %.2f\n", du, dv); // CHECK-EXEC: 1.00 2.00
}
