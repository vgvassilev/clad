// RUN: %cladclang %s -I%S/../../include -o%t 2>&1
// RUN: %t | %filecheck_prints %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double add3(double x, double y, double z) {
  return x + y + z;
}

double prod3(double x, double y, double z) {
  return x * y * z;
}

int main() {
  auto df_add = clad::differentiate<clad::opts::vector_mode>(add3, "x,y,z");
  double add_dx = 0, add_dy = 0, add_dz = 0;
  df_add.execute(1.0, 2.0, 3.0, &add_dx, &add_dy, &add_dz);
  printf("add_dx=%.2f, add_dy=%.2f, add_dz=%.2f\n", add_dx, add_dy, add_dz);
  // prints: add_dx=1.00, add_dy=1.00, add_dz=1.00

  auto df_prod = clad::differentiate<clad::opts::vector_mode>(prod3, "x,y");
  double prod_dx = 0, prod_dy = 0;
  df_prod.execute(2.0, 3.0, 4.0, &prod_dx, &prod_dy);
  printf("prod_dx=%.2f, prod_dy=%.2f\n", prod_dx, prod_dy);
  // prints: prod_dx=12.00, prod_dy=8.00

  return 0;
}
