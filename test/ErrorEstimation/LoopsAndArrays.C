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

//CHECK: void func_grad(float *p, int n, float *_d_p, int *_d_n, double &_final_error) {
//CHECK-NEXT:     float _d_sum = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     unsigned {{int|long}} p_size = 0;
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, sum);
//CHECK-NEXT:         sum += p[i];
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             _final_error += std::abs(_d_sum * sum * {{.+}});
//CHECK-NEXT:             sum = clad::pop(_t1);
//CHECK-NEXT:             float _r_d0 = _d_sum;
//CHECK-NEXT:             _d_p[i] += _r_d0;
//CHECK-NEXT:             p_size = std::max(p_size, i);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(_d_sum * sum * {{.+}});
//CHECK-NEXT:     int i0 = 0;
//CHECK-NEXT:     for (; i0 <= p_size; i0++)
//CHECK-NEXT:         _final_error += std::abs(_d_p[i0] * p[i0] * {{.+}});
//CHECK-NEXT: }


float func2(float x) {
  float z;
  for (int i = 0; i < 9; i++) {
    float m = x * x;
    z = m + m;
  }
  return z;
}

//CHECK: void func2_grad(float x, float *_d_x, double &_final_error) {
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     float _d_m = 0;
//CHECK-NEXT:     float m = 0;
//CHECK-NEXT:     clad::tape<float> _t2 = {};
//CHECK-NEXT:     float z;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (i = 0; i < 9; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, m) , m = x * x;
//CHECK-NEXT:         clad::push(_t2, z);
//CHECK-NEXT:         z = m + m;
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_z += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             _final_error += std::abs(_d_z * z * {{.+}});
//CHECK-NEXT:             z = clad::pop(_t2);
//CHECK-NEXT:             float _r_d0 = _d_z;
//CHECK-NEXT:             _d_z -= _r_d0;
//CHECK-NEXT:             _d_m += _r_d0;
//CHECK-NEXT:             _d_m += _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             _final_error += std::abs(_d_m * m * {{.+}});
//CHECK-NEXT:             *_d_x += _d_m * x;
//CHECK-NEXT:             *_d_x += x * _d_m;
//CHECK-NEXT:             _d_m = 0;
//CHECK-NEXT:             m = clad::pop(_t1);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT: }

float func3(float x, float y) {
  double arr[3];
  arr[0] = x + y;
  arr[1] = x * x;
  arr[2] = arr[0] + arr[1];
  return arr[2];
}

//CHECK: void func3_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     clad::array<double> _d_arr({{3U|3UL}});
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     double _t2;
//CHECK-NEXT:     double arr[3];
//CHECK-NEXT:     _t0 = arr[0];
//CHECK-NEXT:     arr[0] = x + y;
//CHECK-NEXT:     _t1 = arr[1];
//CHECK-NEXT:     arr[1] = x * x;
//CHECK-NEXT:     _t2 = arr[2];
//CHECK-NEXT:     arr[2] = arr[0] + arr[1];
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_arr[2] += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(_d_arr[2] * arr[2] * {{.+}});
//CHECK-NEXT:         arr[2] = _t2;
//CHECK-NEXT:         double _r_d2 = _d_arr[2];
//CHECK-NEXT:         _d_arr[2] -= _r_d2;
//CHECK-NEXT:         _d_arr[0] += _r_d2;
//CHECK-NEXT:         _d_arr[1] += _r_d2;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(_d_arr[1] * arr[1] * {{.+}});
//CHECK-NEXT:         arr[1] = _t1;
//CHECK-NEXT:         double _r_d1 = _d_arr[1];
//CHECK-NEXT:         _d_arr[1] -= _r_d1;
//CHECK-NEXT:         *_d_x += _r_d1 * x;
//CHECK-NEXT:         *_d_x += x * _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(_d_arr[0] * arr[0] * {{.+}});
//CHECK-NEXT:         arr[0] = _t0;
//CHECK-NEXT:         double _r_d0 = _d_arr[0];
//CHECK-NEXT:         _d_arr[0] -= _r_d0;
//CHECK-NEXT:         *_d_x += _r_d0;
//CHECK-NEXT:         *_d_y += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT: }

float func4(float x[10], float y[10]) {
  float sum = 0;
  for (int i = 0; i < 10; i++) {
    x[i] += y[i];
    sum += x[i];
  }
  return sum;
}

//CHECK: void func4_grad(float x[10], float y[10], float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     float _d_sum = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     unsigned {{int|long}} x_size = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     unsigned {{int|long}} y_size = 0;
//CHECK-NEXT:     clad::tape<float> _t2 = {};
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (i = 0; i < 10; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, x[i]);
//CHECK-NEXT:         x[i] += y[i];
//CHECK-NEXT:         clad::push(_t2, sum);
//CHECK-NEXT:         sum += x[i];
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             _final_error += std::abs(_d_sum * sum * {{.+}});
//CHECK-NEXT:             sum = clad::pop(_t2);
//CHECK-NEXT:             float _r_d1 = _d_sum;
//CHECK-NEXT:             _d_x[i] += _r_d1;
//CHECK-NEXT:             x_size = std::max(x_size, i);
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             _final_error += std::abs(_d_x[i] * x[i] * {{.+}});
//CHECK-NEXT:             x[i] = clad::pop(_t1);
//CHECK-NEXT:             float _r_d0 = _d_x[i];
//CHECK-NEXT:             _d_y[i] += _r_d0;
//CHECK-NEXT:             y_size = std::max(y_size, i);
//CHECK-NEXT:             x_size = std::max(x_size, i);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(_d_sum * sum * {{.+}});
//CHECK-NEXT:     int i0 = 0;
//CHECK-NEXT:     for (; i0 <= x_size; i0++)
//CHECK-NEXT:         _final_error += std::abs(_d_x[i0] * x[i0] * {{.+}});
//CHECK-NEXT:     i0 = 0;
//CHECK-NEXT:     for (; i0 <= y_size; i0++)
//CHECK-NEXT:         _final_error += std::abs(_d_y[i0] * y[i0] * {{.+}});
//CHECK-NEXT: }


double func5(double* x, double* y, double* output) {
  output[0] = x[1] * y[2] - x[2] * y[1];
  output[1] = x[2] * y[0] - x[0] * y[2];
  output[2] = x[0] * y[1] - y[0] * x[1];
  return output[0] + output[1] + output[2];
}

//CHECK: void func5_grad(double *x, double *y, double *output, double *_d_x, double *_d_y, double *_d_output, double &_final_error) {
//CHECK-NEXT:     unsigned {{int|long}} output_size = 0;
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     unsigned {{int|long}} x_size = 0;
//CHECK-NEXT:     unsigned {{int|long}} y_size = 0;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     double _t2;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _t0 = output[0];
//CHECK-NEXT:     output[0] = x[1] * y[2] - x[2] * y[1];
//CHECK-NEXT:     _t1 = output[1];
//CHECK-NEXT:     output[1] = x[2] * y[0] - x[0] * y[2];
//CHECK-NEXT:     _t2 = output[2];
//CHECK-NEXT:     output[2] = x[0] * y[1] - y[0] * x[1];
//CHECK-NEXT:     _ret_value0 = output[0] + output[1] + output[2];
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     {
//CHECK-NEXT:         _d_output[0] += 1;
//CHECK-NEXT:         output_size = std::max(output_size, 0);
//CHECK-NEXT:         _d_output[1] += 1;
//CHECK-NEXT:         output_size = std::max(output_size, 1);
//CHECK-NEXT:         _d_output[2] += 1;
//CHECK-NEXT:         output_size = std::max(output_size, 2);
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(_d_output[2] * output[2] * {{.+}});
//CHECK-NEXT:         output[2] = _t2;
//CHECK-NEXT:         double _r_d2 = _d_output[2];
//CHECK-NEXT:         _d_output[2] -= _r_d2;
//CHECK-NEXT:         _d_x[0] += _r_d2 * y[1];
//CHECK-NEXT:         x_size = std::max(x_size, 0);
//CHECK-NEXT:         _d_y[1] += x[0] * _r_d2;
//CHECK-NEXT:         y_size = std::max(y_size, 1);
//CHECK-NEXT:         _d_y[0] += -_r_d2 * x[1];
//CHECK-NEXT:         y_size = std::max(y_size, 0);
//CHECK-NEXT:         _d_x[1] += y[0] * -_r_d2;
//CHECK-NEXT:         x_size = std::max(x_size, 1);
//CHECK-NEXT:         output_size = std::max(output_size, 2);
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(_d_output[1] * output[1] * {{.+}});
//CHECK-NEXT:         output[1] = _t1;
//CHECK-NEXT:         double _r_d1 = _d_output[1];
//CHECK-NEXT:         _d_output[1] -= _r_d1;
//CHECK-NEXT:         _d_x[2] += _r_d1 * y[0];
//CHECK-NEXT:         x_size = std::max(x_size, 2);
//CHECK-NEXT:         _d_y[0] += x[2] * _r_d1;
//CHECK-NEXT:         y_size = std::max(y_size, 0);
//CHECK-NEXT:         _d_x[0] += -_r_d1 * y[2];
//CHECK-NEXT:         x_size = std::max(x_size, 0);
//CHECK-NEXT:         _d_y[2] += x[0] * -_r_d1;
//CHECK-NEXT:         y_size = std::max(y_size, 2);
//CHECK-NEXT:         output_size = std::max(output_size, 1);
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(_d_output[0] * output[0] * {{.+}});
//CHECK-NEXT:         output[0] = _t0;
//CHECK-NEXT:         double _r_d0 = _d_output[0];
//CHECK-NEXT:         _d_output[0] -= _r_d0;
//CHECK-NEXT:         _d_x[1] += _r_d0 * y[2];
//CHECK-NEXT:         x_size = std::max(x_size, 1);
//CHECK-NEXT:         _d_y[2] += x[1] * _r_d0;
//CHECK-NEXT:         y_size = std::max(y_size, 2);
//CHECK-NEXT:         _d_x[2] += -_r_d0 * y[1];
//CHECK-NEXT:         x_size = std::max(x_size, 2);
//CHECK-NEXT:         _d_y[1] += x[2] * -_r_d0;
//CHECK-NEXT:         y_size = std::max(y_size, 1);
//CHECK-NEXT:         output_size = std::max(output_size, 0);
//CHECK-NEXT:     }
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     for (; i <= x_size; i++)
//CHECK-NEXT:         _final_error += std::abs(_d_x[i] * x[i] * {{.+}});
//CHECK-NEXT:     i = 0;
//CHECK-NEXT:     for (; i <= y_size; i++)
//CHECK-NEXT:         _final_error += std::abs(_d_y[i] * y[i] * {{.+}});
//CHECK-NEXT:     i = 0;
//CHECK-NEXT:     for (; i <= output_size; i++)
//CHECK-NEXT:         _final_error += std::abs(_d_output[i] * output[i] * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

int main() {
  clad::estimate_error(func);
  clad::estimate_error(func2);
  clad::estimate_error(func3);
  clad::estimate_error(func4);
  clad::estimate_error(func5);
}
