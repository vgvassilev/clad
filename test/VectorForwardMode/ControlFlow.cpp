// RUN: %cladclang %s -I%S/../../include -o%t 2>&1
// RUN: %t | %filecheck_prints %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double test_if(double x, double y) {
  if (x > y) {
    return x * x;
  } else {
    return y * y;
  }
}

double test_loop(double x, double y) {
  double sum = 0;
  for (int i = 0; i < 3; ++i) {
    sum += x * y;
  }
  return sum;
}

int main() {
  auto df_if = clad::differentiate<clad::opts::vector_mode>(test_if, "x,y");
  
  double dx1 = 0, dy1 = 0;
  df_if.execute(3.0, 2.0, &dx1, &dy1); // x > y -> returns x*x, dx=2x=6, dy=0
  printf("if1: dx=%.2f, dy=%.2f\n", dx1, dy1);
  // prints: if1: dx=6.00, dy=0.00
  
  double dx2 = 0, dy2 = 0;
  df_if.execute(1.0, 4.0, &dx2, &dy2); // x < y -> returns y*y, dx=0, dy=2y=8
  printf("if2: dx=%.2f, dy=%.2f\n", dx2, dy2);
  // prints: if2: dx=0.00, dy=8.00

  auto df_loop = clad::differentiate<clad::opts::vector_mode>(test_loop, "x,y");
  double dx3 = 0, dy3 = 0;
  df_loop.execute(2.0, 3.0, &dx3, &dy3); // sum = 3*x*y, dx=3y=9, dy=3x=6
  printf("loop: dx=%.2f, dy=%.2f\n", dx3, dy3);
  // prints: loop: dx=9.00, dy=6.00

  return 0;
}
