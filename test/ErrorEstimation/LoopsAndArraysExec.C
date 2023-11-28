// RUN: %cladclang %s -I%S/../../include -oLoopsAndArraysExec.out 2>&1 | FileCheck %s
// RUN: ./LoopsAndArraysExec.out | FileCheck -check-prefix=CHECK-EXEC %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

double runningSum(float* f, int n) {
  double sum = 0;
  for (int i = 1; i < n; i++) {
    sum += f[i] + f[i - 1];
  }
  return sum;
}

//CHECK: void runningSum_grad(float *f, int n, clad::array_ref<float> _d_f, clad::array_ref<int> _d_n, double &_final_error) {
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     double _delta_sum = 0;
//CHECK-NEXT:     double _EERepl_sum0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     clad::tape<int> _t3 = {};
//CHECK-NEXT:     clad::tape<double> _EERepl_sum1 = {};
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     _EERepl_sum0 = sum;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 1; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         sum += f[clad::push(_t1, i)] + f[clad::push(_t3, i - 1)];
//CHECK-NEXT:         clad::push(_EERepl_sum1, sum);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             double _r_d0 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d0;
//CHECK-NEXT:             int _t2 = clad::pop(_t1);
//CHECK-NEXT:             _d_f[_t2] += _r_d0;
//CHECK-NEXT:             int _t4 = clad::pop(_t3);
//CHECK-NEXT:             _d_f[_t4] += _r_d0;
//CHECK-NEXT:             double _r0 = clad::pop(_EERepl_sum1);
//CHECK-NEXT:             _delta_sum += std::abs(_r_d0 * _r0 * {{.+}});
//CHECK-NEXT:             _d_sum -= _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_sum += std::abs(_d_sum * _EERepl_sum0 * {{.+}});
//CHECK-NEXT:     clad::array<float> _delta_f(_d_f.size());
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     for (; i < _d_f.size(); i++) {
//CHECK-NEXT:         double _t5 = std::abs(_d_f[i] * f[i] * {{.+}});
//CHECK-NEXT:         _delta_f[i] += _t5;
//CHECK-NEXT:         _final_error += _t5;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += _delta_sum;
//CHECK-NEXT: }

double mulSum(float* a, float* b, int n) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      sum += a[i] * b[j];
  }
  return sum;
}

//CHECK: void mulSum_grad(float *a, float *b, int n, clad::array_ref<float> _d_a, clad::array_ref<float> _d_b, clad::array_ref<int> _d_n, double &_final_error) {
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     double _delta_sum = 0;
//CHECK-NEXT:     double _EERepl_sum0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<unsigned long> _t1 = {};
//CHECK-NEXT:     int _d_j = 0;
//CHECK-NEXT:     clad::tape<float> _t2 = {};
//CHECK-NEXT:     clad::tape<int> _t3 = {};
//CHECK-NEXT:     clad::tape<float> _t5 = {};
//CHECK-NEXT:     clad::tape<int> _t6 = {};
//CHECK-NEXT:     clad::tape<double> _EERepl_sum1 = {};
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     _EERepl_sum0 = sum;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, 0UL);
//CHECK-NEXT:         for (int j = 0; j < n; j++) {
//CHECK-NEXT:             clad::back(_t1)++;
//CHECK-NEXT:             sum += clad::push(_t5, a[clad::push(_t3, i)]) * clad::push(_t2, b[clad::push(_t6, j)]);
//CHECK-NEXT:             clad::push(_EERepl_sum1, sum);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             for (; clad::back(_t1); clad::back(_t1)--) {
//CHECK-NEXT:                 double _r_d0 = _d_sum;
//CHECK-NEXT:                 _d_sum += _r_d0;
//CHECK-NEXT:                 double _r0 = _r_d0 * clad::pop(_t2);
//CHECK-NEXT:                 int _t4 = clad::pop(_t3);
//CHECK-NEXT:                 _d_a[_t4] += _r0;
//CHECK-NEXT:                 double _r1 = clad::pop(_t5) * _r_d0;
//CHECK-NEXT:                 int _t7 = clad::pop(_t6);
//CHECK-NEXT:                 _d_b[_t7] += _r1;
//CHECK-NEXT:                 double _r2 = clad::pop(_EERepl_sum1);
//CHECK-NEXT:                 _delta_sum += std::abs(_r_d0 * _r2 * {{.+}});
//CHECK-NEXT:                 _d_sum -= _r_d0;
//CHECK-NEXT:             }
//CHECK-NEXT:             _d_j = 0;
//CHECK-NEXT:             clad::pop(_t1);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_sum += std::abs(_d_sum * _EERepl_sum0 * {{.+}});
//CHECK-NEXT:     clad::array<float> _delta_a(_d_a.size());
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     for (; i < _d_a.size(); i++) {
//CHECK-NEXT:         double _t8 = std::abs(_d_a[i] * a[i] * {{.+}});
//CHECK-NEXT:         _delta_a[i] += _t8;
//CHECK-NEXT:         _final_error += _t8;
//CHECK-NEXT:     }
//CHECK-NEXT:     clad::array<float> _delta_b(_d_b.size());
//CHECK-NEXT:     i = 0;
//CHECK-NEXT:     for (; i < _d_b.size(); i++) {
//CHECK-NEXT:         double _t9 = std::abs(_d_b[i] * b[i] * {{.+}});
//CHECK-NEXT:         _delta_b[i] += _t9;
//CHECK-NEXT:         _final_error += _t9;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += _delta_sum;
//CHECK-NEXT: }

double divSum(float* a, float* b, int n) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += a[i] / b[i];
  }
  return sum;
}

//CHECK: void divSum_grad(float *a, float *b, int n, clad::array_ref<float> _d_a, clad::array_ref<float> _d_b, clad::array_ref<int> _d_n, double &_final_error) {
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     double _delta_sum = 0;
//CHECK-NEXT:     double _EERepl_sum0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     clad::tape<int> _t2 = {};
//CHECK-NEXT:     clad::tape<float> _t4 = {};
//CHECK-NEXT:     clad::tape<int> _t5 = {};
//CHECK-NEXT:     clad::tape<double> _EERepl_sum1 = {};
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     _EERepl_sum0 = sum;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         sum += clad::push(_t4, a[clad::push(_t2, i)]) / clad::push(_t1, b[clad::push(_t5, i)]);
//CHECK-NEXT:         clad::push(_EERepl_sum1, sum);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             double _r_d0 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d0;
//CHECK-NEXT:             float _r0 = clad::pop(_t1);
//CHECK-NEXT:             double _r1 = _r_d0 / _r0;
//CHECK-NEXT:             int _t3 = clad::pop(_t2);
//CHECK-NEXT:             _d_a[_t3] += _r1;
//CHECK-NEXT:             double _r2 = _r_d0 * -clad::pop(_t4) / (_r0 * _r0);
//CHECK-NEXT:             int _t6 = clad::pop(_t5);
//CHECK-NEXT:             _d_b[_t6] += _r2;
//CHECK-NEXT:             double _r3 = clad::pop(_EERepl_sum1);
//CHECK-NEXT:             _delta_sum += std::abs(_r_d0 * _r3 * {{.+}});
//CHECK-NEXT:             _d_sum -= _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_sum += std::abs(_d_sum * _EERepl_sum0 * {{.+}});
//CHECK-NEXT:     clad::array<float> _delta_a(_d_a.size());
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     for (; i < _d_a.size(); i++) {
//CHECK-NEXT:         double _t7 = std::abs(_d_a[i] * a[i] * {{.+}});
//CHECK-NEXT:         _delta_a[i] += _t7;
//CHECK-NEXT:         _final_error += _t7;
//CHECK-NEXT:     }
//CHECK-NEXT:     clad::array<float> _delta_b(_d_b.size());
//CHECK-NEXT:     i = 0;
//CHECK-NEXT:     for (; i < _d_b.size(); i++) {
//CHECK-NEXT:         double _t8 = std::abs(_d_b[i] * b[i] * {{.+}});
//CHECK-NEXT:         _delta_b[i] += _t8;
//CHECK-NEXT:         _final_error += _t8;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += _delta_sum;
//CHECK-NEXT: }

int main() {
  auto df = clad::estimate_error(runningSum);
  float arrf[3] = {0.456, 0.77, 0.95};
  double finalError = 0;
  float darr[3] = {0, 0, 0};
  int dn = 0;
  clad::array_ref<float> darrRef(darr, 3);
  df.execute(arrf, 3, darrRef, &dn, finalError);
  printf("Result (RS) = {%.2f, %.2f, %.2f} error = %.5f\n", darr[0], darr[1],
         darr[2], finalError); // CHECK-EXEC: Result (RS) = {1.00, 2.00, 1.00} error = 0.00000

  finalError = 0;
  darr[0] = darr[1] = darr[2] = 0;
  dn = 0;
  float darr2[3] = {0, 0, 0};
  clad::array_ref<float> darrRef2(darr2, 3);
  auto df2 = clad::estimate_error(mulSum);
  df2.execute(arrf, arrf, 3, darrRef, darrRef2, &dn, finalError);
  printf("Result (MS) = {%.2f, %.2f, %.2f}, {%.2f, %.2f, %.2f}  error = %.5f\n",
         darr[0], darr[1], darr[2], darr2[0], darr2[1], darr2[2],
         finalError); // CHECK-EXEC: Result (MS) = {2.18, 2.18, 2.18}, {2.18, 2.18, 2.18}  error = 0.00000

  finalError = 0;
  darr[0] = darr[1] = darr[2] = 0;
  darr2[0] = darr2[1] = darr2[2] = 0;
  dn = 0;
  auto df3 = clad::estimate_error(divSum);
  df3.execute(arrf, arrf, 3, darrRef, darrRef2, &dn, finalError);
  printf("Result (DS) = {%.2f, %.2f, %.2f}, {%.2f, %.2f, %.2f}  error = %.5f\n",
         darr[0], darr[1], darr[2], darr2[0], darr2[1], darr2[2],
         finalError); // CHECK-EXEC: Result (DS) = {2.19, 1.30, 1.05}, {-2.19, -1.30, -1.05}  error = 0.00000
}
