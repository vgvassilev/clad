// RUN: %cladclang %s -I%S/../../include -oEnzymeLoops.out 2>&1 | FileCheck %s
// RUN: ./EnzymeLoops.out | FileCheck -check-prefix=CHECK-EXEC %s
// REQUIRES: Enzyme
// CHECK-NOT: {{.*error|warning|note:.*}}


#include "clad/Differentiator/Differentiator.h"
#include <cmath>

#include "../TestUtils.h"

double f1(double x) {
  double t = 1;
  for (int i = 0; i < 3; i++)
    t *= x;
  return t;
} // == x^3

// CHECK: void f1_grad_enzyme(double x, double *_d_x) {
// CHECK-NEXT:     clad::EnzymeGradient<1> grad = __enzyme_autodiff_f1(f1, x);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT: }

double f2(double x) {
  double t = 1;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      t *= x;
  return t;
} // == x^9

// CHECK: void f2_grad_enzyme(double x, double *_d_x) {
// CHECK-NEXT:     clad::EnzymeGradient<1> grad = __enzyme_autodiff_f2(f2, x);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT: }

double f3(double x) {
  double t = 1;
  for (int i = 0; i < 3; i++) {
    t *= x;
    if (i == 1)
      return t;
  }
  return t;
} // == x^2

// CHECK: void f3_grad_enzyme(double x, double *_d_x) {
// CHECK-NEXT:     clad::EnzymeGradient<1> grad = __enzyme_autodiff_f3(f3, x);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT: }

double f4(double x) {
  double t = 1;
  for (int i = 0; i < 3; t *= x)
    i++;
  return t;
} // == x^3

// CHECK: void f4_grad_enzyme(double x, double *_d_x) {
// CHECK-NEXT:     clad::EnzymeGradient<1> grad = __enzyme_autodiff_f4(f4, x);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT: }

double f5(double x){
  for (int i = 0; i < 10; i++)
    x++;
  return x;
} // == x + 10

// CHECK: void f5_grad_enzyme(double x, double *_d_x) {
// CHECK-NEXT:     clad::EnzymeGradient<1> grad = __enzyme_autodiff_f5(f5, x);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT: }

double f6 (double i, double j) {
  double a = 0;
  for (int counter=0; counter<3; ++counter) {
    double b = i*i;
    double c = j*j;
    b += j;
    a += b + c + i;
  }
  return a;
}

// CHECK: void f6_grad_enzyme(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f6(f6, i, j);
// CHECK-NEXT:     *_d_i = grad.d_arr[0U];
// CHECK-NEXT:     *_d_j = grad.d_arr[1U];
// CHECK-NEXT: }

double fn7(double i, double j) {
  double a = 0;
  int counter = 3;
  while (counter--)
    a += i*i + j;
  return a;
}

// CHECK: void fn7_grad_enzyme(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_fn7(fn7, i, j);
// CHECK-NEXT:     *_d_i = grad.d_arr[0U];
// CHECK-NEXT:     *_d_j = grad.d_arr[1U];
// CHECK-NEXT: }

double fn8(double i, double j) {
  double a = 0;
  int counter = 3;
  while (counter > 0)
    do {
      a += i*i + j;
    } while (--counter);
  return a;
}

// CHECK: void fn8_grad_enzyme(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_fn8(fn8, i, j);
// CHECK-NEXT:     *_d_i = grad.d_arr[0U];
// CHECK-NEXT:     *_d_j = grad.d_arr[1U];
// CHECK-NEXT: }

double fn9(double i, double j) {
  int counter, counter_again;
  counter = counter_again = 3;
  double a = 0;
  while (counter--) {
    counter_again = 3;
    while (counter_again--) {
      a += i*i + j;
    }
  }
  return a;
}

// CHECK: void fn9_grad_enzyme(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_fn9(fn9, i, j);
// CHECK-NEXT:     *_d_i = grad.d_arr[0U];
// CHECK-NEXT:     *_d_j = grad.d_arr[1U];
// CHECK-NEXT: }

double fn10(double i, double j) {
  int counter = 3;
  double a = 0;
  do {
    a += i*i + j;
    counter -= 1;
  } while (counter);
  return a;
}

// CHECK: void fn10_grad_enzyme(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_fn10(fn10, i, j);
// CHECK-NEXT:     *_d_i = grad.d_arr[0U];
// CHECK-NEXT:     *_d_j = grad.d_arr[1U];
// CHECK-NEXT: }

double fn11(double i, double j) {
  int counter = 3;
  double a = 0;
  do {
    int counter_again = 3;
    do {
      a += i*i + j;
      counter_again -= 1;
      do
        a += j;
      while (0);
    } while (counter_again);
    counter -= 1;
  } while (counter);
  return a;
}

// CHECK: void fn11_grad_enzyme(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_fn11(fn11, i, j);
// CHECK-NEXT:     *_d_i = grad.d_arr[0U];
// CHECK-NEXT:     *_d_j = grad.d_arr[1U];
// CHECK-NEXT: }

double fn12(double i, double j) {
  int counter = 5;
  double res=0;
  for (int ii=0; ii<counter; ++ii) {
    if (ii == 4) {
      res += i*j;
      break;
    }
    if (ii > 2) {
      res += 2*i;
      continue;
    }
    res += i + j;
  }
  return res;
}

// CHECK: void fn12_grad_enzyme(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_fn12(fn12, i, j);
// CHECK-NEXT:     *_d_i = grad.d_arr[0U];
// CHECK-NEXT:     *_d_j = grad.d_arr[1U];
// CHECK-NEXT: }

double fn13(double i, double j) {
  int counter = 5;
  double res = 0;
  for (int ii=0; ii<counter; ++ii) {
    int jj = ii;
    if (ii < 2)
      continue;
    while (jj--) {
      if (jj < 3) {
        res += i*j;
        break;
      } else {
        continue;
      }
      res += i*i*j*j;
    }
  }
  return res;
}

// CHECK: void fn13_grad_enzyme(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_fn13(fn13, i, j);
// CHECK-NEXT:     *_d_i = grad.d_arr[0U];
// CHECK-NEXT:     *_d_j = grad.d_arr[1U];
// CHECK-NEXT: }

double fn14(double i, double j) {
  int choice = 5;
  double res = 0;
  for (int counter=0; counter<choice; ++counter)
    if (counter < 2)
      res += i+j;
    else if (counter < 4)
      continue;
    else {
      res += 2*i + 2*j;
      break;
    }
  return res;
}

// CHECK: void fn14_grad_enzyme(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_fn14(fn14, i, j);
// CHECK-NEXT:     *_d_i = grad.d_arr[0U];
// CHECK-NEXT:     *_d_j = grad.d_arr[1U];
// CHECK-NEXT: }

#define TEST(F, x) { \
  resultEnzyme[0] = 0; \
  resultClad[0] = 0; \
  auto F##gradEnzyme = clad::gradient<clad::opts::use_enzyme>(F);\
  F##gradEnzyme.execute(x, &resultEnzyme[0]);\
  printf("{%.2f}\n", resultEnzyme[0]); \
  auto F##gradClad = clad::gradient(F);                                           \
  F##gradClad.execute(x, &resultClad[0]);                      \
  test_utils::EssentiallyEqual(resultClad[0],resultEnzyme[0]); \
}

#define TEST_2(F, x, y)                                                        \
  {                                                                            \
    resultEnzyme[0] = resultEnzyme[1] = 0;                                     \
    resultClad[0] = resultClad[1] = 0;                                         \
    auto F##gradEnzyme = clad::gradient<clad::opts::use_enzyme>(F);            \
    F##gradEnzyme.execute(x, y, resultEnzyme, resultEnzyme + 1);               \
    printf("{%.2f, %.2f}\n", resultEnzyme[0], resultEnzyme[1]);                \
    auto F##gradClad = clad::gradient(F);                                      \
    F##gradClad.execute(x, y, resultClad, resultClad+1);                       \
    test_utils::EssentiallyEqualArrays(resultClad,resultEnzyme,2);             \
  }


int main() {
  double resultEnzyme[5] = {};
  double resultClad[5] = {};
  TEST(f1, 3); // CHECK-EXEC: {27.00}
  TEST(f2, 3); // CHECK-EXEC: {59049.00}
  TEST(f3, 3); // CHECK-EXEC: {6.00}
  TEST(f4, 3); // CHECK-EXEC: {27.00}
  TEST(f5, 3); // CHECK-EXEC: {1.00}

  TEST_2(f6, 3, 5);       // CHECK-EXEC: {21.00, 33.00}
  TEST_2(fn7, 3, 5);      // CHECK-EXEC: {18.00, 3.00}
  TEST_2(fn8, 3, 5);      // CHECK-EXEC: {18.00, 3.00}
  TEST_2(fn9, 3, 5);      // CHECK-EXEC: {54.00, 9.00}
  TEST_2(fn10, 3, 5);     // CHECK-EXEC: {18.00, 3.00}
  TEST_2(fn11, 3, 5);     // CHECK-EXEC: {54.00, 18.00}
  TEST_2(fn12, 3, 5);     // CHECK-EXEC: {10.00, 6.00}
  TEST_2(fn13, 3, 5);     // CHECK-EXEC: {15.00, 9.00}
  TEST_2(fn14, 3, 5);     // CHECK-EXEC: {4.00, 4.00}
}
