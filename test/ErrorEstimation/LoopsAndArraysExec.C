// RUN: %cladclang %s -I%S/../../include -oLoopsAndArraysExec.out 2>&1 | %filecheck %s
// RUN: ./LoopsAndArraysExec.out | %filecheck_exec %s
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

#include <cmath>

double runningSum(float* f, int n) {
  double sum = 0;
  for (int i = 1; i < n; i++) {
    sum += f[i] + f[i - 1];
  }
  return sum;
}

//CHECK: void runningSum_grad(float *f, int n, float *_d_f, int *_d_n, double &_final_error) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     unsigned {{int|long}} f_size = {{0U|0UL}};
//CHECK-NEXT:     double _d_sum = 0.;
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = 0;
//CHECK-NEXT:     for (i = 1; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, sum);
//CHECK-NEXT:         sum += f[i] + f[i - 1];
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             _final_error += std::abs(_d_sum * sum * {{.+}});
//CHECK-NEXT:             sum = clad::pop(_t1);
//CHECK-NEXT:             double _r_d0 = _d_sum;
//CHECK-NEXT:             _d_f[i] += _r_d0;
//CHECK-NEXT:             f_size = std::max(f_size, i);
//CHECK-NEXT:             _d_f[i - 1] += _r_d0;
//CHECK-NEXT:             f_size = std::max(f_size, i - 1);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(_d_sum * sum * {{.+}});
//CHECK-NEXT:     int i0 = 0;
//CHECK-NEXT:     for (; i0 <= f_size; i0++)
//CHECK-NEXT:         _final_error += std::abs(_d_f[i0] * f[i0] * {{.+}});
//CHECK-NEXT: }

double mulSum(float* a, float* b, int n) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      sum += a[i] * b[j];
  }
  return sum;
}

//CHECK: void mulSum_grad(float *a, float *b, int n, float *_d_a, float *_d_b, int *_d_n, double &_final_error) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<unsigned {{int|long|long long}}> _t1 = {};
//CHECK-NEXT:     clad::tape<int> _t2 = {};
//CHECK-NEXT:     int _d_j = 0;
//CHECK-NEXT:     int j = 0;
//CHECK-NEXT:     clad::tape<double> _t3 = {};
//CHECK-NEXT:     unsigned {{int|long}} b_size = {{0U|0UL}};
//CHECK-NEXT:     unsigned {{int|long}} a_size = {{0U|0UL}};
//CHECK-NEXT:     double _d_sum = 0.;
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = 0;
//CHECK-NEXT:     for (i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, 0);
//CHECK-NEXT:         for (clad::push(_t2, j) , j = 0; j < n; j++) {
//CHECK-NEXT:             clad::back(_t1)++;
//CHECK-NEXT:             clad::push(_t3, sum);
//CHECK-NEXT:             sum += a[i] * b[j];
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             for (; clad::back(_t1); clad::back(_t1)--) {
//CHECK-NEXT:                 j--;
//CHECK-NEXT:                 _final_error += std::abs(_d_sum * sum * {{.+}});
//CHECK-NEXT:                 sum = clad::pop(_t3);
//CHECK-NEXT:                 double _r_d0 = _d_sum;
//CHECK-NEXT:                 b_size = std::max(b_size, j);
//CHECK-NEXT:                 _d_a[i] += _r_d0 * b[j];
//CHECK-NEXT:                 a_size = std::max(a_size, i);
//CHECK-NEXT:                 _d_b[j] += a[i] * _r_d0;
//CHECK-NEXT:                 b_size = std::max(b_size, j);
//CHECK-NEXT:             }
//CHECK-NEXT:             {
//CHECK-NEXT:                 _d_j = 0;
//CHECK-NEXT:                 j = clad::pop(_t2);
//CHECK-NEXT:             }
//CHECK-NEXT:             clad::pop(_t1);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(_d_sum * sum * {{.+}});
//CHECK-NEXT:     int i0 = 0;
//CHECK-NEXT:     for (; i0 <= a_size; i0++)
//CHECK-NEXT:         _final_error += std::abs(_d_a[i0] * a[i0] * {{.+}});
//CHECK-NEXT:     i0 = 0;
//CHECK-NEXT:     for (; i0 <= b_size; i0++)
//CHECK-NEXT:         _final_error += std::abs(_d_b[i0] * b[i0] * {{.+}});
//CHECK-NEXT: }

double divSum(float* a, float* b, int n) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += a[i] / b[i];
  }
  return sum;
}

//CHECK: void divSum_grad(float *a, float *b, int n, float *_d_a, float *_d_b, int *_d_n, double &_final_error) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     unsigned {{int|long}} b_size = {{0U|0UL}};
//CHECK-NEXT:     unsigned {{int|long}} a_size = {{0U|0UL}};
//CHECK-NEXT:     double _d_sum = 0.;
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = 0;
//CHECK-NEXT:     for (i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, sum);
//CHECK-NEXT:         sum += a[i] / b[i];
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             _final_error += std::abs(_d_sum * sum * {{.+}});
//CHECK-NEXT:             sum = clad::pop(_t1);
//CHECK-NEXT:             double _r_d0 = _d_sum;
//CHECK-NEXT:             b_size = std::max(b_size, i);
//CHECK-NEXT:             _d_a[i] += _r_d0 / b[i];
//CHECK-NEXT:             a_size = std::max(a_size, i);
//CHECK-NEXT:             double _r0 = _r_d0 * -(a[i] / (b[i] * b[i]));
//CHECK-NEXT:             _d_b[i] += _r0;
//CHECK-NEXT:             b_size = std::max(b_size, i);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(_d_sum * sum * {{.+}});
//CHECK-NEXT:     int i0 = 0;
//CHECK-NEXT:     for (; i0 <= a_size; i0++)
//CHECK-NEXT:         _final_error += std::abs(_d_a[i0] * a[i0] * {{.+}});
//CHECK-NEXT:     i0 = 0;
//CHECK-NEXT:     for (; i0 <= b_size; i0++)
//CHECK-NEXT:         _final_error += std::abs(_d_b[i0] * b[i0] * {{.+}});
//CHECK-NEXT: }

int main() {
  float arrf[3] = {0.456, 0.77, 0.95};
  float darr[3] = {0, 0, 0};
  int dn = 0;
  INIT_ERROR_ESTIMATION(runningSum);
  TEST_ERROR_ESTIMATION(runningSum, /*PrecissionType*/float, arrf, 3, darr, &dn); // CHECK-EXEC: {7.12}
  printf("Result (RS) = {%.2f, %.2f, %.2f}\n", darr[0], darr[1], darr[2]); 
         // CHECK-EXEC: Result (RS) = {1.00, 2.00, 1.00}

  darr[0] = darr[1] = darr[2] = 0;
  dn = 0;
  float darr2[3] = {0, 0, 0};
  INIT_ERROR_ESTIMATION(mulSum);
  TEST_ERROR_ESTIMATION(mulSum, /*PrecissionType*/float, arrf, arrf, 3, darr, darr2, &dn); // CHECK-EXEC: {28.85}
  printf("Result (MS) = {%.2f, %.2f, %.2f}, {%.2f, %.2f, %.2f}\n",
         darr[0], darr[1], darr[2], darr2[0], darr2[1], darr2[2]);
         // CHECK-EXEC: Result (MS) = {2.18, 2.18, 2.18}, {2.18, 2.18, 2.18}

  darr[0] = darr[1] = darr[2] = 0;
  darr2[0] = darr2[1] = darr2[2] = 0;
  dn = 0;
  INIT_ERROR_ESTIMATION(divSum);
  TEST_ERROR_ESTIMATION(divSum, /*PrecissionType*/float, arrf, arrf, 3, darr, darr2, &dn); // CHECK-EXEC: {12.00}
  printf("Result (DS) = {%.2f, %.2f, %.2f}, {%.2f, %.2f, %.2f}\n",
         darr[0], darr[1], darr[2], darr2[0], darr2[1], darr2[2]);
         // CHECK-EXEC: Result (DS) = {2.19, 1.30, 1.05}, {-2.19, -1.30, -1.05}
}
