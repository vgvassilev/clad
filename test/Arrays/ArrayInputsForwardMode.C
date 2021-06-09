// RUN: %cladclang %s -lm -I%S/../../include -oArrayInputsForwardMode.out 2>&1 | FileCheck %s
// RUN: ./ArrayInputsForwardMode.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double multiply(double *arr) {
  return arr[0] * arr[1];
}

//CHECK:   double multiply_darg0_1(double *arr) {
//CHECK-NEXT:       return 0 * arr[1] + arr[0] * 1;
//CHECK-NEXT:   }

double divide(double *arr) {
  return arr[0] / arr[1];
}

//CHECK:   double divide_darg0_1(double *arr) {
//CHECK_NEXT:       return (0 * arr[1] - arr[0] * 1) / (arr[1] * arr[1]);
//CHECK_NEXT:   }

double addArr(double *arr, int n) {
  double ret = 0;
  for (int i = 0; i < n; i++) {
    ret += arr[i];
  }
  return ret;
}

//CHECK:   double addArr_darg0_1(double *arr, int n) {
//CHECK-NEXT:       int _d_n = 0;
//CHECK-NEXT:       double _d_ret = 0;
//CHECK-NEXT:       double ret = 0;
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < n; i++) {
//CHECK-NEXT:               _d_ret += (i == 1);
//CHECK-NEXT:               ret += arr[i];
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_ret;
//CHECK-NEXT:   }

int main() {
  double arr[] = {1, 2, 3, 4, 5};
  auto multiply_dx = clad::differentiate(multiply, "arr[1]");
  printf("Result = {%.2f}\n", multiply_dx.execute(arr)); // CHECK-EXEC: Result = {1.00}

  auto divide_dx = clad::differentiate(divide, "arr[1]");
  printf("Result = {%.2f}\n", divide_dx.execute(arr)); // CHECK-EXEC: Result = {-0.25}

  auto addArr_dx = clad::differentiate(addArr, "arr[1]");
  printf("Result = {%.2f}\n", addArr_dx.execute(arr, 5)); // CHECK-EXEC: Result = {1.00}
}
