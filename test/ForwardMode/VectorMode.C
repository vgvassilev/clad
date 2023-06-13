// RUN: %cladclang %s -lm -lstdc++ -I%S/../../include -oVectorMode.out 2>&1 | FileCheck %s
// RUN: ./VectorMode.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f1(double x, double y) {
  return x*y*(x+y);
}

void f1_d_all_args(double x, double y, double *_d_x, double *_d_y);

// CHECK: void f1_d_all_args(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:   clad::array<double> _d_vector_x = {1., 0.};
// CHECK-NEXT:   clad::array<double> _d_vector_y = {0., 1.};
// CHECK-NEXT:   double _t0 = x * y;
// CHECK-NEXT:   double _t1 = (x + y);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return = (_d_vector_x * y + x * _d_vector_y) * _t1 + _t0 * (_d_vector_x + _d_vector_y);
// CHECK-NEXT:     *_d_x = _d_vector_return[0];
// CHECK-NEXT:     *_d_y = _d_vector_return[1];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double f2(double x, double y) {
  // to test usage of local variables.
  double temp1 = x*y;
  double temp2 = x+y;
  return temp1*temp2;
}

void f2_d_all_args(double x, double y, double *_d_x, double *_d_y);

// CHECK: void f2_d_all_args(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:   clad::array<double> _d_vector_x = {1., 0.};
// CHECK-NEXT:   clad::array<double> _d_vector_y = {0., 1.};
// CHECK-NEXT:   clad::array<double> _d_vector_temp1 = _d_vector_x * y + x * _d_vector_y;
// CHECK-NEXT:   double temp1 = x * y;
// CHECK-NEXT:   clad::array<double> _d_vector_temp2 = _d_vector_x + _d_vector_y;
// CHECK-NEXT:   double temp2 = x + y;
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return = _d_vector_temp1 * temp2 + temp1 * _d_vector_temp2;
// CHECK-NEXT:     *_d_x = _d_vector_return[0];
// CHECK-NEXT:     *_d_y = _d_vector_return[1];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double f3(double x, double y) {
  // x * abs(y)
  if (y < 0) // to test if statements.
    y = -y;
  return x*y;
}

void f3_d_all_args(double x, double y, double *_d_x, double *_d_y);

// CHECK: void f3_d_all_args(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:   clad::array<double> _d_vector_x = {1., 0.};
// CHECK-NEXT:   clad::array<double> _d_vector_y = {0., 1.};
// CHECK-NEXT:   if (y < 0) {
// CHECK-NEXT:     _d_vector_y = - _d_vector_y;
// CHECK-NEXT:     y = -y;
// CHECK-NEXT:   }
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return = _d_vector_x * y + x * _d_vector_y;
// CHECK-NEXT:     *_d_x = _d_vector_return[0];
// CHECK-NEXT:     *_d_y = _d_vector_return[1];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

#define TEST(F, x, y)                                                          \
  {                                                                            \
    result[0] = 0;                                                             \
    result[1] = 0;                                                             \
    clad::vector_forward_differentiate(F);                                     \
    F##_d_all_args(x, y, &result[0], &result[1]);                              \
    printf("Result is = {%.2f, %.2f}\n", result[0], result[1]);                \
  }

int main() {
  double result[2];

  TEST(f1, 3, 4); // CHECK-EXEC: Result is = {40.00, 33.00}
  TEST(f2, 3, 4); // CHECK-EXEC: Result is = {40.00, 33.00}
  TEST(f3, 3, -4); // CHECK-EXEC: Result is = {4.00, -3.00}
}
