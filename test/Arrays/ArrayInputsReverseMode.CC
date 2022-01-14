// RUN: %cladclang %s -lm -lstdc++ -I%S/../../include -oArrayInputsReverseMode.out 2>&1 | FileCheck %s
// RUN: ./ArrayInputsReverseMode.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double addArr(double *arr, int n) {
  double ret = 0;
  for (int i = 0; i < n; i++) {
    ret += arr[i];
  }
  return ret;
}

//CHECK:   void addArr_grad(double *arr, int n, clad::array_ref<double> _d_arr, clad::array_ref<double> _d_n) {
//CHECK-NEXT:       double _d_ret = 0;
//CHECK-NEXT:       unsigned long _t0;
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       clad::tape<int> _t1 = {};
//CHECK-NEXT:       double ret = 0;
//CHECK-NEXT:       _t0 = 0;
//CHECK-NEXT:       for (int i = 0; i < n; i++) {
//CHECK-NEXT:           _t0++;
//CHECK-NEXT:           ret += arr[clad::push(_t1, i)];
//CHECK-NEXT:       }
//CHECK-NEXT:       double addArr_return = ret;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_ret += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d0 = _d_ret;
//CHECK-NEXT:               _d_ret += _r_d0;
//CHECK-NEXT:               _d_arr[clad::pop(_t1)] += _r_d0;
//CHECK-NEXT:               _d_ret -= _r_d0;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f(double *arr) {
  return addArr(arr, 3);
}

//CHECK:   void f_grad(double *arr, clad::array_ref<double> _d_arr) {
//CHECK-NEXT:       double *_t0;
//CHECK-NEXT:       _t0 = arr;
//CHECK-NEXT:       double f_return = addArr(_t0, 3);
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           clad::array<double> _grad0(_d_arr.size());
//CHECK-NEXT:           double _grad1 = 0.;
//CHECK-NEXT:           addArr_grad(_t0, 3, _grad0, &_grad1);
//CHECK-NEXT:           clad::array_ref<double> _r0(_grad0 *= 1);
//CHECK-NEXT:           _d_arr += _r0;
//CHECK-NEXT:           double _r1 = 1 * _grad1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

int main() {
  double arr[] = {1, 2, 3};
  auto f_dx = clad::gradient(f);

  double darr[3] = {};
  clad::array_ref<double> darr_ref(darr, 3);
  f_dx.execute(arr, darr_ref);

  printf("Result = {%.2f, %.2f, %.2f}\n", darr[0], darr[1], darr[2]); // CHECK-EXEC: Result = {1.00, 1.00, 1.00}
}
