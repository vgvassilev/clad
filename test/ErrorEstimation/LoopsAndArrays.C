// RUN: %cladclang %s -x c++ -lstdc++ -I%S/../../include -oLoopsAndArrays.out 2>&1 | FileCheck %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <iostream>

// Arrays in loops
float func(float* p, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += p[i];
  }
  return sum;
}

// CHECK: void func_grad(float *p, int n, float *_result, double &_final_error) {
// CHECK-NEXT:     double _delta_sum = 0;
// CHECK-NEXT:     float _EERepl_sum0;
// CHECK-NEXT:     float _d_sum = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     clad::tape<float> _EERepl_sum1 = {};
// CHECK-NEXT:     float sum = 0;
// CHECK-NEXT:     _EERepl_sum0 = sum;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int i = 0; i < n; i++) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         sum += p[clad::push(_t1, i)];
// CHECK-NEXT:         clad::push(_EERepl_sum1, sum);
// CHECK-NEXT:     }
// CHECK-NEXT:     float func_return = sum;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_sum += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             float _r_d0 = _d_sum;
// CHECK-NEXT:             _d_sum += _r_d0;
// CHECK-NEXT:             _result[clad::pop(_t1)] += _r_d0;
// CHECK-NEXT:             float _r0 = clad::pop(_EERepl_sum1);
// CHECK-NEXT:             _delta_sum += _r_d0 * _r0 * {{.+}};
// CHECK-NEXT:             _d_sum -= _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     _delta_sum += _d_sum * _EERepl_sum0 * {{.+}};
// CHECK-NEXT:     _final_error += _delta_sum;
// CHECK-NEXT: }

float func2(float x) {
  float z;
  for (int i = 0; i < 9; i++) {
    float m = x * x;
    z = m + m;
  }
  return z;
}

// CHECK: void func2_grad(float x, float *_result, double &_final_error) {
// CHECK-NEXT:     double _delta_z = 0;
// CHECK-NEXT:     float _EERepl_z0;
// CHECK-NEXT:     float _d_z = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     clad::tape<float> _t1 = {};
// CHECK-NEXT:     clad::tape<float> _t2 = {};
// CHECK-NEXT:     double _delta_m = 0;
// CHECK-NEXT:     clad::tape<float> _EERepl_m0 = {};
// CHECK-NEXT:     float _d_m = 0;
// CHECK-NEXT:     clad::tape<float> _EERepl_z1 = {};
// CHECK-NEXT:     float z;
// CHECK-NEXT:     _EERepl_z0 = z;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int i = 0; i < 9; i++) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         float m = clad::push(_t2, x) * clad::push(_t1, x);
// CHECK-NEXT:         clad::push(_EERepl_m0, m);
// CHECK-NEXT:         z = m + m;
// CHECK-NEXT:         clad::push(_EERepl_z1, z);
// CHECK-NEXT:     }
// CHECK-NEXT:     float func2_return = z;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_z += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             float _r_d0 = _d_z;
// CHECK-NEXT:             _d_m += _r_d0;
// CHECK-NEXT:             _d_m += _r_d0;
// CHECK-NEXT:             float _r3 = clad::pop(_EERepl_z1);
// CHECK-NEXT:             _delta_z += _r_d0 * _r3 * {{.+}};
// CHECK-NEXT:             _d_z -= _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             float _r0 = _d_m * clad::pop(_t1);
// CHECK-NEXT:             _result[0UL] += _r0;
// CHECK-NEXT:             float _r1 = clad::pop(_t2) * _d_m;
// CHECK-NEXT:             _result[0UL] += _r1;
// CHECK-NEXT:             float _r2 = clad::pop(_EERepl_m0);
// CHECK-NEXT:             _delta_m += _d_m * _r2 * {{.+}};
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     double _delta_x = 0;
// CHECK-NEXT:     _delta_x += _result[0UL] * x * {{.+}};
// CHECK-NEXT:     _final_error += _delta_{{x|z|m}} + _delta_{{x|z|m}} + _delta_{{x|z|m}};
// CHECK-NEXT: }

float func3(float x, float y) {
  double arr[3];
  arr[0] = x + y;
  arr[1] = x * x;
  arr[2] = arr[0] + arr[1];
  return arr[2];
}

// CHECK: void func3_grad(float x, float y, float *_result, double &_final_error) {
// CHECK-NEXT:     double _delta_arr[3] = {};
// CHECK-NEXT:     double _d_arr[3] = {};
// CHECK-NEXT:     double _EERepl_arr0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     double _EERepl_arr1;
// CHECK-NEXT:     double _EERepl_arr2;
// CHECK-NEXT:     double arr[3];
// CHECK-NEXT:     arr[0] = x + y;
// CHECK-NEXT:     _EERepl_arr0 = arr[0];
// CHECK-NEXT:     _t1 = x;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     arr[1] = _t1 * _t0;
// CHECK-NEXT:     _EERepl_arr1 = arr[1];
// CHECK-NEXT:     arr[2] = arr[0] + arr[1];
// CHECK-NEXT:     _EERepl_arr2 = arr[2];
// CHECK-NEXT:     double func3_return = arr[2];
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_arr[2] += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d2 = _d_arr[2];
// CHECK-NEXT:         _d_arr[0] += _r_d2;
// CHECK-NEXT:         _d_arr[1] += _r_d2;
// CHECK-NEXT:         _delta_arr[2] += _r_d2 * _EERepl_arr2 * {{.+}};
// CHECK-NEXT:         _final_error += _delta_arr[2];
// CHECK-NEXT:         _d_arr[2] -= _r_d2;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = _d_arr[1];
// CHECK-NEXT:         double _r0 = _r_d1 * _t0;
// CHECK-NEXT:         _result[0UL] += _r0;
// CHECK-NEXT:         double _r1 = _t1 * _r_d1;
// CHECK-NEXT:         _result[0UL] += _r1;
// CHECK-NEXT:         _delta_arr[1] += _r_d1 * _EERepl_arr1 * {{.+}};
// CHECK-NEXT:         _final_error += _delta_arr[1];
// CHECK-NEXT:         _d_arr[1] -= _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_arr[0];
// CHECK-NEXT:         _result[0UL] += _r_d0;
// CHECK-NEXT:         _result[1UL] += _r_d0;
// CHECK-NEXT:         _delta_arr[0] += _r_d0 * _EERepl_arr0 * {{.+}};
// CHECK-NEXT:         _final_error += _delta_arr[0];
// CHECK-NEXT:         _d_arr[0] -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     double _delta_x = 0;
// CHECK-NEXT:     _delta_x += _result[0UL] * x * {{.+}};
// CHECK-NEXT:     double _delta_y = 0;
// CHECK-NEXT:     _delta_y += _result[1UL] * y * {{.+}};
// CHECK-NEXT:     _final_error += _delta_{{y|x}} + _delta_{{y|x}};
// CHECK-NEXT: }

int main() {

  clad::estimate_error(func);
  clad::estimate_error(func2);
  clad::estimate_error(func3);
}
