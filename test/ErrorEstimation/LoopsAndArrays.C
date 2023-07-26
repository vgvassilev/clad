// RUN: %cladclang -I%S/../../include -oLoopsAndArrays.out %s 2>&1 | FileCheck %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

// Arrays in loops
float func(float* p, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += p[i];
  }
  return sum;
}

//CHECK: void func_grad(float *p, int n, clad::array_ref<float> _d_p, clad::array_ref<int> _d_n, double &_final_error) {
//CHECK-NEXT:     float _d_sum = 0;
//CHECK-NEXT:     double _delta_sum = 0;
//CHECK-NEXT:     float _EERepl_sum0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     clad::tape<float> _EERepl_sum1 = {};
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     _EERepl_sum0 = sum;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         sum += p[clad::push(_t1, i)];
//CHECK-NEXT:         clad::push(_EERepl_sum1, sum);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             float _r_d0 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d0;
//CHECK-NEXT:             int _t2 = clad::pop(_t1);
//CHECK-NEXT:             _d_p[_t2] += _r_d0;
//CHECK-NEXT:             float _r0 = clad::pop(_EERepl_sum1);
//CHECK-NEXT:             _delta_sum += std::abs(_r_d0 * _r0 * {{.+}});
//CHECK-NEXT:             _d_sum -= _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_sum += std::abs(_d_sum * _EERepl_sum0 * {{.+}});
//CHECK-NEXT:     clad::array<float> _delta_p(_d_p.size());
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     for (; i < _d_p.size(); i++) {
//CHECK-NEXT:         double _t3 = std::abs(_d_p[i] * p[i] * {{.+}});
//CHECK-NEXT:         _delta_p[i] += _t3;
//CHECK-NEXT:         _final_error += _t3;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += _delta_sum;
//CHECK-NEXT: }


float func2(float x) {
  float z;
  for (int i = 0; i < 9; i++) {
    float m = x * x;
    z = m + m;
  }
  return z;
}

//CHECK: void func2_grad(float x, clad::array_ref<float> _d_x, double &_final_error) {
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     double _delta_z = 0;
//CHECK-NEXT:     float _EERepl_z0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     clad::tape<float> _t2 = {};
//CHECK-NEXT:     float _d_m = 0;
//CHECK-NEXT:     double _delta_m = 0;
//CHECK-NEXT:     clad::tape<float> _EERepl_m0 = {};
//CHECK-NEXT:     clad::tape<float> _EERepl_z1 = {};
//CHECK-NEXT:     float z;
//CHECK-NEXT:     _EERepl_z0 = z;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < 9; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         float m = clad::push(_t2, x) * clad::push(_t1, x);
//CHECK-NEXT:         clad::push(_EERepl_m0, m);
//CHECK-NEXT:         z = m + m;
//CHECK-NEXT:         clad::push(_EERepl_z1, z);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_z += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             float _r_d0 = _d_z;
//CHECK-NEXT:             _d_m += _r_d0;
//CHECK-NEXT:             _d_m += _r_d0;
//CHECK-NEXT:             float _r3 = clad::pop(_EERepl_z1);
//CHECK-NEXT:             _delta_z += std::abs(_r_d0 * _r3 * {{.+}});
//CHECK-NEXT:             _d_z -= _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             float _r0 = _d_m * clad::pop(_t1);
//CHECK-NEXT:             * _d_x += _r0;
//CHECK-NEXT:             float _r1 = clad::pop(_t2) * _d_m;
//CHECK-NEXT:             * _d_x += _r1;
//CHECK-NEXT:             _d_m = 0;
//CHECK-NEXT:             float _r2 = clad::pop(_EERepl_m0);
//CHECK-NEXT:             _delta_m += std::abs(_d_m * _r2 * {{.+}});
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|z|m}} + _delta_{{x|z|m}} + _delta_{{x|z|m}};
//CHECK-NEXT: }

float func3(float x, float y) {
  double arr[3];
  arr[0] = x + y;
  arr[1] = x * x;
  arr[2] = arr[0] + arr[1];
  return arr[2];
}

//CHECK: void func3_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     clad::array<double> _d_arr(3UL);
//CHECK-NEXT:     clad::array<double> _delta_arr(_d_arr.size());
//CHECK-NEXT:     double _EERepl_arr0;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     double _EERepl_arr1;
//CHECK-NEXT:     double _EERepl_arr2;
//CHECK-NEXT:     double arr[3];
//CHECK-NEXT:     arr[0] = x + y;
//CHECK-NEXT:     _EERepl_arr0 = arr[0];
//CHECK-NEXT:     _t1 = x;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     arr[1] = _t1 * _t0;
//CHECK-NEXT:     _EERepl_arr1 = arr[1];
//CHECK-NEXT:     arr[2] = arr[0] + arr[1];
//CHECK-NEXT:     _EERepl_arr2 = arr[2];
//CHECK-NEXT:     _d_arr[2] += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d2 = _d_arr[2];
//CHECK-NEXT:         _d_arr[0] += _r_d2;
//CHECK-NEXT:         _d_arr[1] += _r_d2;
//CHECK-NEXT:         _delta_arr[2] += std::abs(_r_d2 * _EERepl_arr2 * {{.+}});
//CHECK-NEXT:         _final_error += _delta_arr[2];
//CHECK-NEXT:         _d_arr[2] -= _r_d2;
//CHECK-NEXT:         _d_arr[2];
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d1 = _d_arr[1];
//CHECK-NEXT:         double _r0 = _r_d1 * _t0;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:         double _r1 = _t1 * _r_d1;
//CHECK-NEXT:         * _d_x += _r1;
//CHECK-NEXT:         _delta_arr[1] += std::abs(_r_d1 * _EERepl_arr1 * {{.+}});
//CHECK-NEXT:         _final_error += _delta_arr[1];
//CHECK-NEXT:         _d_arr[1] -= _r_d1;
//CHECK-NEXT:         _d_arr[1];
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d0 = _d_arr[0];
//CHECK-NEXT:         * _d_x += _r_d0;
//CHECK-NEXT:         * _d_y += _r_d0;
//CHECK-NEXT:         _delta_arr[0] += std::abs(_r_d0 * _EERepl_arr0 * {{.+}});
//CHECK-NEXT:         _final_error += _delta_arr[0];
//CHECK-NEXT:         _d_arr[0] -= _r_d0;
//CHECK-NEXT:         _d_arr[0];
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{y|x}} + _delta_{{y|x}};
//CHECK-NEXT: }

float func4(float x[10], float y[10]) {
  float sum = 0;
  for (int i = 0; i < 10; i++) {
    x[i] += y[i];
    sum += x[i];
  }
  return sum;
}

//CHECK: void func4_grad(float x[10], float y[10], clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     float _d_sum = 0;
//CHECK-NEXT:     double _delta_sum = 0;
//CHECK-NEXT:     float _EERepl_sum0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     clad::array<float> _delta_x(_d_x.size());
//CHECK-NEXT:     clad::array<float> _EERepl_x0(_d_x.size());
//CHECK-NEXT:     for (int i0 = 0; i0 < _d_x.size(); i0++) {
//CHECK-NEXT:         _EERepl_x0[i0] = x[i0];
//CHECK-NEXT:     }
//CHECK-NEXT:     clad::tape<int> _t3 = {};
//CHECK-NEXT:     clad::tape<float> _EERepl_x1 = {};
//CHECK-NEXT:     clad::tape<int> _t5 = {};
//CHECK-NEXT:     clad::tape<float> _EERepl_sum1 = {};
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     _EERepl_sum0 = sum;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < 10; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         x[clad::push(_t1, i)] += y[clad::push(_t3, i)];
//CHECK-NEXT:         clad::push(_EERepl_x1, x[clad::push(_t1, i)]);
//CHECK-NEXT:         sum += x[clad::push(_t5, i)];
//CHECK-NEXT:         clad::push(_EERepl_sum1, sum);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             float _r_d1 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d1;
//CHECK-NEXT:             int _t6 = clad::pop(_t5);
//CHECK-NEXT:             _d_x[_t6] += _r_d1;
//CHECK-NEXT:             float _r2 = clad::pop(_EERepl_sum1);
//CHECK-NEXT:             _delta_sum += std::abs(_r_d1 * _r2 * {{.+}});
//CHECK-NEXT:             _d_sum -= _r_d1;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             int _t2 = clad::pop(_t1);
//CHECK-NEXT:             float _r_d0 = _d_x[_t2];
//CHECK-NEXT:             _d_x[_t2] += _r_d0;
//CHECK-NEXT:             int _t4 = clad::pop(_t3);
//CHECK-NEXT:             _d_y[_t4] += _r_d0;
//CHECK-NEXT:             float _r0 = clad::pop(_EERepl_x1);
//CHECK-NEXT:             int _r1 = clad::pop(_t1);
//CHECK-NEXT:             _delta_x[_r1] += std::abs(_r_d0 * _r0 * {{.+}});
//CHECK-NEXT:             _final_error += _delta_x[_r1];
//CHECK-NEXT:             _d_x[_t2] -= _r_d0;
//CHECK-NEXT:             _d_x[_t2];
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_sum += std::abs(_d_sum * _EERepl_sum0 * {{.+}});
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     for (; i < _d_x.size(); i++) {
//CHECK-NEXT:         double _t7 = std::abs(_d_x[i] * _EERepl_x0[i] * {{.+}});
//CHECK-NEXT:         _delta_x[i] += _t7;
//CHECK-NEXT:         _final_error += _t7;
//CHECK-NEXT:     }
//CHECK-NEXT:     clad::array<float> _delta_y(_d_y.size());
//CHECK-NEXT:     i = 0;
//CHECK-NEXT:     for (; i < _d_y.size(); i++) {
//CHECK-NEXT:         double _t8 = std::abs(_d_y[i] * y[i] * {{.+}});
//CHECK-NEXT:         _delta_y[i] += _t8;
//CHECK-NEXT:         _final_error += _t8;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += _delta_sum;
//CHECK-NEXT: }


double func5(double* x, double* y, double* output) {
  output[0] = x[1] * y[2] - x[2] * y[1];
  output[1] = x[2] * y[0] - x[0] * y[2];
  output[2] = x[0] * y[1] - y[0] * x[1];
  return output[0] + output[1] + output[2];
}

//CHECK: void func5_grad(double *x, double *y, double *output, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y, clad::array_ref<double> _d_output, double &_final_error) {
//CHECK-NEXT:     clad::array<double> _delta_output(_d_output.size());
//CHECK-NEXT:     clad::array<double> _EERepl_output0(_d_output.size());
//CHECK-NEXT:     for (int i = 0; i < _d_output.size(); i++) {
//CHECK-NEXT:         _EERepl_output0[i] = output[i];
//CHECK-NEXT:     }
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     double _t2;
//CHECK-NEXT:     double _t3;
//CHECK-NEXT:     double _EERepl_output1;
//CHECK-NEXT:     double _t4;
//CHECK-NEXT:     double _t5;
//CHECK-NEXT:     double _t6;
//CHECK-NEXT:     double _t7;
//CHECK-NEXT:     double _EERepl_output2;
//CHECK-NEXT:     double _t8;
//CHECK-NEXT:     double _t9;
//CHECK-NEXT:     double _t10;
//CHECK-NEXT:     double _t11;
//CHECK-NEXT:     double _EERepl_output3;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _t1 = x[1];
//CHECK-NEXT:     _t0 = y[2];
//CHECK-NEXT:     _t3 = x[2];
//CHECK-NEXT:     _t2 = y[1];
//CHECK-NEXT:     output[0] = _t1 * _t0 - _t3 * _t2;
//CHECK-NEXT:     _EERepl_output1 = output[0];
//CHECK-NEXT:     _t5 = x[2];
//CHECK-NEXT:     _t4 = y[0];
//CHECK-NEXT:     _t7 = x[0];
//CHECK-NEXT:     _t6 = y[2];
//CHECK-NEXT:     output[1] = _t5 * _t4 - _t7 * _t6;
//CHECK-NEXT:     _EERepl_output2 = output[1];
//CHECK-NEXT:     _t9 = x[0];
//CHECK-NEXT:     _t8 = y[1];
//CHECK-NEXT:     _t11 = y[0];
//CHECK-NEXT:     _t10 = x[1];
//CHECK-NEXT:     output[2] = _t9 * _t8 - _t11 * _t10;
//CHECK-NEXT:     _EERepl_output3 = output[2];
//CHECK-NEXT:     _ret_value0 = output[0] + output[1] + output[2];
//CHECK-NEXT:     {
//CHECK-NEXT:         _d_output[0] += 1;
//CHECK-NEXT:         _d_output[1] += 1;
//CHECK-NEXT:         _d_output[2] += 1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d2 = _d_output[2];
//CHECK-NEXT:         double _r8 = _r_d2 * _t8;
//CHECK-NEXT:         _d_x[0] += _r8;
//CHECK-NEXT:         double _r9 = _t9 * _r_d2;
//CHECK-NEXT:         _d_y[1] += _r9;
//CHECK-NEXT:         double _r10 = -_r_d2 * _t10;
//CHECK-NEXT:         _d_y[0] += _r10;
//CHECK-NEXT:         double _r11 = _t11 * -_r_d2;
//CHECK-NEXT:         _d_x[1] += _r11;
//CHECK-NEXT:         _delta_output[2] += std::abs(_r_d2 * _EERepl_output3 * {{.+}});
//CHECK-NEXT:         _final_error += _delta_output[2];
//CHECK-NEXT:         _d_output[2] -= _r_d2;
//CHECK-NEXT:         _d_output[2];
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d1 = _d_output[1];
//CHECK-NEXT:         double _r4 = _r_d1 * _t4;
//CHECK-NEXT:         _d_x[2] += _r4;
//CHECK-NEXT:         double _r5 = _t5 * _r_d1;
//CHECK-NEXT:         _d_y[0] += _r5;
//CHECK-NEXT:         double _r6 = -_r_d1 * _t6;
//CHECK-NEXT:         _d_x[0] += _r6;
//CHECK-NEXT:         double _r7 = _t7 * -_r_d1;
//CHECK-NEXT:         _d_y[2] += _r7;
//CHECK-NEXT:         _delta_output[1] += std::abs(_r_d1 * _EERepl_output2 * {{.+}});
//CHECK-NEXT:         _final_error += _delta_output[1];
//CHECK-NEXT:         _d_output[1] -= _r_d1;
//CHECK-NEXT:         _d_output[1];
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d0 = _d_output[0];
//CHECK-NEXT:         double _r0 = _r_d0 * _t0;
//CHECK-NEXT:         _d_x[1] += _r0;
//CHECK-NEXT:         double _r1 = _t1 * _r_d0;
//CHECK-NEXT:         _d_y[2] += _r1;
//CHECK-NEXT:         double _r2 = -_r_d0 * _t2;
//CHECK-NEXT:         _d_x[2] += _r2;
//CHECK-NEXT:         double _r3 = _t3 * -_r_d0;
//CHECK-NEXT:         _d_y[1] += _r3;
//CHECK-NEXT:         _delta_output[0] += std::abs(_r_d0 * _EERepl_output1 * {{.+}});
//CHECK-NEXT:         _final_error += _delta_output[0];
//CHECK-NEXT:         _d_output[0] -= _r_d0;
//CHECK-NEXT:         _d_output[0];
//CHECK-NEXT:     }
//CHECK-NEXT:     clad::array<double> _delta_x(_d_x.size());
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     for (; i < _d_x.size(); i++) {
//CHECK-NEXT:         double _t12 = std::abs(_d_x[i] * x[i] * {{.+}});
//CHECK-NEXT:         _delta_x[i] += _t12;
//CHECK-NEXT:         _final_error += _t12;
//CHECK-NEXT:     }
//CHECK-NEXT:     clad::array<double> _delta_y(_d_y.size());
//CHECK-NEXT:     i = 0;
//CHECK-NEXT:     for (; i < _d_y.size(); i++) {
//CHECK-NEXT:         double _t13 = std::abs(_d_y[i] * y[i] * {{.+}});
//CHECK-NEXT:         _delta_y[i] += _t13;
//CHECK-NEXT:         _final_error += _t13;
//CHECK-NEXT:     }
//CHECK-NEXT:     i = 0;
//CHECK-NEXT:     for (; i < _d_output.size(); i++) {
//CHECK-NEXT:         double _t14 = std::abs(_d_output[i] * _EERepl_output0[i] * {{.+}});
//CHECK-NEXT:         _delta_output[i] += _t14;
//CHECK-NEXT:         _final_error += _t14;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

int main() {
  clad::estimate_error(func);
  clad::estimate_error(func2);
  clad::estimate_error(func3);
  clad::estimate_error(func4);
  clad::estimate_error(func5);
}
