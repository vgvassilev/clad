// RUN: %cladclang %s  -I%S/../../include -oEnzymeGradients.out 2>&1 | FileCheck %s
// RUN: ./EnzymeGradients.out | FileCheck -check-prefix=CHECK-EXEC %s
// REQUIRES: Enzyme
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

#include "../TestUtils.h"

__attribute__((always_inline)) double f_add1(double x, double y) {
  return x + y;
}

// CHECK: void f_add1_grad_enzyme(double x, double y, double *_d_x, double *_d_y) __attribute__((always_inline)) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_add1(f_add1, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_add2(double x, double y) {
  return 3*x + 4*y;
}

// CHECK: void f_add2_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_add2(f_add2, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_add3(double x, double y) {
  return 3*x + 4*y*4;
}

// CHECK: void f_add3_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_add3(f_add3, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_sub1(double x, double y) {
  return x - y;
}

// CHECK: void f_sub1_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_sub1(f_sub1, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_sub2(double x, double y) {
  return 3*x - 4*y;
}

// CHECK: void f_sub2_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_sub2(f_sub2, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_mult1(double x, double y) {
  return x*y;
}

// CHECK: void f_mult1_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_mult1(f_mult1, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_mult2(double x, double y) {
   return 3*x*4*y;
}

// CHECK: void f_mult2_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_mult2(f_mult2, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_div1(double x, double y) {
  return x/y;
}

// CHECK: void f_div1_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_div1(f_div1, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_div2(double x, double y) {
  return 3*x/(4*y);
}

// CHECK: void f_div2_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_div2(f_div2, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_c(double x, double y) {
  return -x*y + (x + y)*(x/y) - x*x; 
}

// CHECK: void f_c_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_c(f_c, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_rosenbrock(double x, double y) {
  return (x - 1) * (x - 1) + 100 * (y - x * x) * (y - x * x);
}

// CHECK: void f_rosenbrock_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_rosenbrock(f_rosenbrock, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_cond1(double x, double y) {
  return (x > y ? x : y);
}

// CHECK: void f_cond1_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_cond1(f_cond1, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_cond2(double x, double y) {
  return (x > y ? x : (y > 0 ? y : -y));
}

// CHECK: void f_cond2_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_cond2(f_cond2, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_cond3(double x, double c) {
  return (c > 0 ? x + c : x - c);
}

// CHECK: void f_cond3_grad_enzyme(double x, double c, double *_d_x, double *_d_c) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_cond3(f_cond3, x, c);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_c = grad.d_arr[1U];
// CHECK-NEXT: }

double f_if1(double x, double y) {
  if (x > y)
    return x;
  else
    return y;
}

// CHECK: void f_if1_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_if1(f_if1, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_if2(double x, double y) {
  if (x > y)
    return x;
  else if (y > 0)
    return y;
  else
    return -y;
}

// CHECK: void f_if2_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_if2(f_if2, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_decls1(double x, double y) {
  double a = 3 * x;
  double b = 5 * y;
  double c = a + b;
  return 2 * c;
}

// CHECK: void f_decls1_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_decls1(f_decls1, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_decls2(double x, double y) {
  double a = x * x;
  double b = x * y;
  double c = y * y;
  return a + 2 * b + c;
}

// CHECK: void f_decls2_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_decls2(f_decls2, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_decls3(double x, double y) {
  double a = 3 * x;
  double c = 333 * y;
  if (x > 1)
    return 2 * a;
  else if (x < -1)
    return -2 * a;
  double b = a * a;
  return b;
}

// CHECK: void f_decls3_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_decls3(f_decls3, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_issue138(double x, double y) {
    double _t1 = 1; // expect it not to collide with _t*
    return x*x*x*x + y*y*y*y;
}

// CHECK: void f_issue138_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_issue138(f_issue138, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

double f_const(const double a, const double b) {
  return a * b;
}

// CHECK: void f_const_grad_enzyme(const double a, const double b, double *_d_a, double *_d_b) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f_const(f_const, a, b);
// CHECK-NEXT:     *_d_a = grad.d_arr[0U];
// CHECK-NEXT:     *_d_b = grad.d_arr[1U];
// CHECK-NEXT: }

#define TEST(F, x, y)                                                            \
  {                                                                              \
    resultEnzyme[0] = 0;                                                         \
    resultEnzyme[1] = 0;                                                         \
    resultClad[0] = 0;                                                           \
    resultClad[1] = 0;                                                           \
    auto F##gradEnzyme = clad::gradient<clad::opts::use_enzyme>(F);                 \
    F##gradEnzyme.execute(x, y, &resultEnzyme[0], &resultEnzyme[1]);                \
    printf("Result is = {%.2f, %.2f}\n", resultEnzyme[0], resultEnzyme[1]);      \
    auto F##gradClad = clad::gradient(F);                                           \
    F##gradClad.execute(x, y, &resultClad[0], &resultClad[1]);                      \
    test_utils::EssentiallyEqualArrays(resultClad,resultEnzyme,2);               \
  }

int main() {
  double resultEnzyme[2];
  double resultClad[2];

  TEST(f_add1, 1, 1); // CHECK-EXEC: Result is = {1.00, 1.00}
  TEST(f_add2, 1, 1); // CHECK-EXEC: Result is = {3.00, 4.00}
  TEST(f_add3, 1, 1); // CHECK-EXEC: Result is = {3.00, 16.00}
  TEST(f_sub1, 1, 1); // CHECK-EXEC: Result is = {1.00, -1.00}
  TEST(f_sub2, 1, 1); // CHECK-EXEC: Result is = {3.00, -4.00}
  TEST(f_mult1, 1, 1); // CHECK-EXEC: Result is = {1.00, 1.00}
  TEST(f_mult2, 1, 1); // CHECK-EXEC: Result is = {12.00, 12.00}
  TEST(f_div1, 1, 1); // CHECK-EXEC: Result is = {1.00, -1.00}
  TEST(f_div2, 1, 1); // CHECK-EXEC: Result is = {0.75, -0.75}
  TEST(f_c, 1, 1); // CHECK-EXEC: Result is = {0.00, -2.00}
  TEST(f_rosenbrock, 1, 1); // CHECK-EXEC: Result is = {0.00, 0.00}
  TEST(f_cond1, 3, 2); // CHECK-EXEC: Result is = {1.00, 0.00}
  TEST(f_cond2, 3, -1); // CHECK-EXEC: Result is = {1.00, 0.00}
  TEST(f_cond3, 3, -1); // CHECK-EXEC: Result is = {1.00, -1.00}
  TEST(f_if1, 3, 2); // CHECK-EXEC: Result is = {1.00, 0.00}
  TEST(f_if2, -5, -4); // CHECK-EXEC: Result is = {0.00, -1.00}
  TEST(f_decls1, 3, 3); // CHECK-EXEC: Result is = {6.00, 10.00}
  TEST(f_decls2, 2, 2); // CHECK-EXEC: Result is = {8.00, 8.00}
  TEST(f_decls3, 3, 0); // CHECK-EXEC: Result is = {6.00, 0.00}
  TEST(f_decls3, -3, 0); // CHECK-EXEC: Result is = {-6.00, 0.00}
  TEST(f_decls3, 0.5, 0); // CHECK-EXEC: Result is = {9.00, 0.00}
  TEST(f_decls3, 0, 100); // CHECK-EXEC: Result is = {0.00, 0.00}
  TEST(f_issue138, 1, 2); // CHECK-EXEC: Result is = {4.00, 32.00}
  TEST(f_const, 2, 3); // CHECK-EXEC: Result is = {3.00, 2.00}
}
