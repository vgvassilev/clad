// RUN: %cladclang %s -I%S/../../include -oArrayInputsForwardMode.out 2>&1 | %filecheck %s
// RUN: ./ArrayInputsForwardMode.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double multiply(const double *arr) {
  return arr[0] * arr[1];
}

//CHECK:   double multiply_darg0_1(const double *arr) {
//CHECK-NEXT:       return 0 * arr[1] + arr[0] * 1.;
//CHECK-NEXT:   }

double divide(const double *arr) {
  return arr[0] / arr[1];
}

//CHECK:   double divide_darg0_1(const double *arr) {
//CHECK-NEXT:       return (0 * arr[1] - arr[0] * 1.) / (arr[1] * arr[1]);
//CHECK-NEXT:   }

double addArr(const double *arr, int n) {
  double ret = 0;
  for (int i = 0; i < n; i++) {
    ret += arr[i];
  }
  return ret;
}

//CHECK:   double addArr_darg0_1(const double *arr, int n) {
//CHECK-NEXT:       int _d_n = 0;
//CHECK-NEXT:       double _d_ret = 0;
//CHECK-NEXT:       double ret = 0;
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < n; i++) {
//CHECK-NEXT:               _d_ret += (i == 1.);
//CHECK-NEXT:               ret += arr[i];
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_ret;
//CHECK-NEXT:   }

double numMultIndex(double* arr, size_t n, double x) {
  // compute x * i, where arr[i] = x
  // if x is not present in arr, return 0
  bool flag = false;
  size_t idx = 0;
  for (size_t i = 0; i < n; ++i) {
    if (arr[i] == x) {
      flag = true;
      idx = i;
      break;
    }
  }
  return flag ? idx * x : 0;
}

// CHECK:   double numMultIndex_darg2(double *arr, size_t n, double x) {
// CHECK-NEXT:     size_t _d_n = 0;
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_flag = 0;
// CHECK-NEXT:     bool flag = false;
// CHECK-NEXT:     size_t _d_idx = 0;
// CHECK-NEXT:     size_t idx = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         size_t _d_i = 0;
// CHECK-NEXT:         for (size_t i = 0; i < n; ++i) {
// CHECK-NEXT:             if (arr[i] == x) {
// CHECK-NEXT:                 _d_flag = 0;
// CHECK-NEXT:                 flag = true;
// CHECK-NEXT:                 _d_idx = _d_i;
// CHECK-NEXT:                 idx = i;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return flag ? _d_idx * x + idx * _d_x : 0;
// CHECK-NEXT: }

int main() {
  double arr[] = {1, 2, 3, 4, 5};
  auto multiply_dx = clad::differentiate(multiply, "arr[1]");
  printf("Result = {%.2f}\n", multiply_dx.execute(arr)); // CHECK-EXEC: Result = {1.00}

  auto divide_dx = clad::differentiate(divide, "arr[1]");
  printf("Result = {%.2f}\n", divide_dx.execute(arr)); // CHECK-EXEC: Result = {-0.25}

  auto addArr_dx = clad::differentiate(addArr, "arr[1]");
  printf("Result = {%.2f}\n", addArr_dx.execute(arr, 5)); // CHECK-EXEC: Result = {1.00}

  auto numMultIndex_dx = clad::differentiate(numMultIndex, "x");
  printf("Result = {%.2f}\n", numMultIndex_dx.execute(arr, 5, 4)); // CHECK-EXEC: Result = {3.00}
}
