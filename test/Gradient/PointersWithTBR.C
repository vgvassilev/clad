// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oPointersWithTBR.out
// RUN: ./PointersWithTBR.out | %filecheck_exec %s
// XFAIL: *
// CHECK-NOT: {{.*error|warning|note:.*}}

// FIXME: This is currently marked XFAIL because of lack of pointer support in
// TBR analysis. Once TBR supports pointers, this test should be enabled
// and possibily combined with Pointer.C

#include "clad/Differentiator/Differentiator.h"

double minimalPointer(double x) {
  double* const p = &x;
  *p = (*p)*(*p);
  return *p; // x*x
}

double pointerParam(const double* arr, size_t n) {
  double sum = 0;
  for (size_t i=0; i < n; ++i) {
    size_t* j = &i;
    sum += arr[0] * (*j);
    arr = arr + 1;
  }
  return sum;
}

int main() {
    auto d_minimalPointer = clad::gradient(minimalPointer, "x");
    double x = 5;
    double dx = 0;
    d_minimalPointer.execute(x, &dx);
    printf("%.2f\n", dx); // CHECK-EXEC: 10.00

    auto d_pointerParam = clad::gradient(pointerParam, "arr");
    double arr[5] = {1, 2, 3, 4, 5};
    double d_arr[5] = {0, 0, 0, 0, 0};
    d_pointerParam.execute(arr, 5, d_arr);
    printf("%.2f %.2f %.2f %.2f %.2f\n", d_arr[0], d_arr[1], d_arr[2], d_arr[3], d_arr[4]); // CHECK-EXEC: 0.00 1.00 2.00 3.00 4.00
}
