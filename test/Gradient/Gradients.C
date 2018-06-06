// RUN: %cladclang %s -I%S/../../include -oGradients.out 2>&1 | FileCheck %s
// RUN: ./Gradients.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f_add1(double x, double y) {
  return x + y;
}

// CHECK: void f_add1_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += 1.;
// CHECK-NEXT:   _result[1UL] += 1.;
// CHECK-NEXT: }
void f_add1_grad(double x, double y, double *_result);

double f_add2(double x, double y) {
  return 3*x + 4*y;
}

// CHECK: void f_add2_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += 3 * 1.;
// CHECK-NEXT:   _result[1UL] += 4 * 1.;
// CHECK-NEXT: }
void f_add2_grad(double x, double y, double *_result);

double f_add3(double x, double y) {
  return 3*x + 4*y*4;
}

// CHECK: void f_add3_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += 3 * 1.;
// CHECK-NEXT:   _result[1UL] += 4 * 1. * 4;
// CHECK-NEXT: }
void f_add3_grad(double x, double y, double *_result);

double f_sub1(double x, double y) {
  return x - y;
}

// CHECK: void f_sub1_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += 1.;
// CHECK-NEXT:   _result[1UL] += -1.;
// CHECK-NEXT: }
void f_sub1_grad(double x, double y, double *_result);

double f_sub2(double x, double y) {
  return 3*x - 4*y;
}

// CHECK: void f_sub2_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += 3 * 1.;
// CHECK-NEXT:   _result[1UL] += 4 * -1.;
// CHECK-NEXT: }
void f_sub2_grad(double x, double y, double *_result);

double f_mult1(double x, double y) {
  return x*y;
}

// CHECK: void f_mult1_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += 1. * y;
// CHECK-NEXT:   _result[1UL] += x * 1.;
// CHECK-NEXT: }
void f_mult1_grad(double x, double y, double *_result);

double f_mult2(double x, double y) {
   return 3*x*4*y;
}

// CHECK: void f_mult2_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += 3 * 1. * y * 4;
// CHECK-NEXT:   _result[1UL] += 3 * x * 4 * 1.;
// CHECK-NEXT: }
void f_mult2_grad(double x, double y, double *_result);

double f_div1(double x, double y) {
  return x/y;
}

// CHECK: void f_div1_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += 1. / y;
// CHECK-NEXT:   _result[1UL] += -x / y * y;
// CHECK-NEXT: }
void f_div1_grad(double x, double y, double *_result);

double f_div2(double x, double y) {
  return 3*x/(4*y);
}

// CHECK: void f_div2_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += 3 * 1. / (4 * y);
// CHECK-NEXT:   _result[1UL] += 4 * -3 * x / (4 * y) * (4 * y);
// CHECK-NEXT: }
void f_div2_grad(double x, double y, double *_result);

double f_c(double x, double y) {
  return -x*y + (x + y)*(x/y) - x*x; 
}

// CHECK: void f_c_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += -1. * y;
// CHECK-NEXT:   _result[1UL] += -x * 1.;
// CHECK-NEXT:   _result[0UL] += 1. * (x / y);
// CHECK-NEXT:   _result[1UL] += 1. * (x / y);
// CHECK-NEXT:   _result[0UL] += 1. / y;
// CHECK-NEXT:   _result[1UL] += -x / y * y;
// CHECK-NEXT:   _result[0UL] += -1. * x;
// CHECK-NEXT:   _result[0UL] += x * -1.;
// CHECK-NEXT: }
void f_c_grad(double x, double y, double *_result);

double f_rosenbrock(double x, double y) {
  return (x - 1) * (x - 1) + 100 * (y - x * x) * (y - x * x);
}

// CHECK: void f_rosenbrock_grad(double x, double y, double *_result) {
// CHECK-NEXT:   _result[0UL] += 1. * (x - 1);
// CHECK-NEXT:   _result[0UL] += (x - 1) * 1.;
// CHECK-NEXT:   _result[1UL] += 100 * 1. * (y - x * x);
// CHECK-NEXT:   _result[0UL] += -100 * 1. * (y - x * x) * x;
// CHECK-NEXT:   _result[0UL] += x * -100 * 1. * (y - x * x);
// CHECK-NEXT:   _result[1UL] += 100 * (y - x * x) * 1.;
// CHECK-NEXT:   _result[0UL] += -100 * (y - x * x) * 1. * x;
// CHECK-NEXT:   _result[0UL] += x * -100 * (y - x * x) * 1.;
// CHECK-NEXT: }
void f_rosenbrock_grad(double x, double y, double *_result);

unsigned f_types(int x, float y, double z) {
  return x + y + z;
}

// CHECK: void f_types_grad(int x, float y, double z, unsigned int *_result) {
// CHECK-NEXT:   _result[0UL] += 1U;
// CHECK-NEXT:   _result[1UL] += 1U;
// CHECK-NEXT:   _result[2UL] += 1U;
// CHECK-NEXT: }
void f_types_grad(int x, float y, double z, unsigned int *_result);

#define TEST(F, x, y) { \
  result[0] = 0; result[1] = 0;\
  clad::gradient(F);\
  F##_grad(x, y, result);\
  printf("Result is = {%.2f, %.2f}\n", result[0], result[1]); \
}
 
int main() { // expected-no-diagnostics
  double result[2];

  TEST(f_add1, 1, 1); // CHECK-EXEC: Result is = {1.00, 1.00}
  TEST(f_add2, 1, 1); // CHECK-EXEC: Result is = {3.00, 4.00}
  TEST(f_add3, 1, 1); // CHECK-EXEC: Result is = {3.00, 16.00}
  TEST(f_sub1, 1, 1); // CHECK-EXEC: Result is = {1.00, -1.00}
  TEST(f_sub2, 1, 1); // CHECK-EXEC: Result is = {3.00, -4.00}
  TEST(f_mult1, 1, 1); // CHECK-EXEC: Result is = {1.00, 1.00}
  TEST(f_mult2, 1, 1); // CHECK-EXEC: Result is = {12.00, 12.00}
  TEST(f_div1, 1, 1); // CHECK-EXEC: Result is = {1.00, -1.00}
  TEST(f_div2, 1, 1); // CHECK-EXEC: Result is = {0.75, -0.75}
  TEST(f_c, 1, 1); // CHECK-EXEC: Result is = {-1.00, -1.00}
  TEST(f_rosenbrock, 1, 1); // CHECK-EXEC: Result is = {0.00, 0.00}
  clad::gradient(f_types);
}

