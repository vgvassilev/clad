// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | %filecheck %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

// Add/Sub operations
float func(float x, float y) {
  x = x + y;
  y = y + (y++) + y; // expected-warning {{unsequenced modification and access to 'y'}}
  float z = y * x;
  return z;
}

//CHECK: void func_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     float _t0 = x;
//CHECK-NEXT:     x = x + y;
//CHECK-NEXT:     float _t1 = y;
//CHECK-NEXT:     y = y + y++ + y;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     float z = y * x;
//CHECK-NEXT:     _d_z += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(_d_z * z * {{.+}});
//CHECK-NEXT:         *_d_y += _d_z * x;
//CHECK-NEXT:         *_d_x += y * _d_z;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:         _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:         y = _t1;
//CHECK-NEXT:         float _r_d1 = *_d_y;
//CHECK-NEXT:         *_d_y = 0;
//CHECK-NEXT:         *_d_y += _r_d1;
//CHECK-NEXT:         *_d_y += _r_d1;
//CHECK-NEXT:         y--;
//CHECK-NEXT:         *_d_y += _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         float _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x = 0;
//CHECK-NEXT:         *_d_x += _r_d0;
//CHECK-NEXT:         *_d_y += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT: }

// This function may evaluate incorrectly due to absence of usage of
// absolute values
float func2(float x, float y) {
  x = x - y - y * y;
  float z = y / x;
  return z;
}

//CHECK: void func2_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     float _t0 = x;
//CHECK-NEXT:     x = x - y - y * y;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     float z = y / x;
//CHECK-NEXT:     _d_z += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(_d_z * z * {{.+}});
//CHECK-NEXT:         *_d_y += _d_z / x;
//CHECK-NEXT:         float _r0 = _d_z * -(y / (x * x));
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         float _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x = 0;
//CHECK-NEXT:         *_d_x += _r_d0;
//CHECK-NEXT:         *_d_y += -_r_d0;
//CHECK-NEXT:         *_d_y += -_r_d0 * y;
//CHECK-NEXT:         *_d_y += y * -_r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT: }


// Subexpression assign
float func3(float x, float y) {
  x = x - y - y * y;
  float z = y;
  float t = x * z * (y = x + x);
  return t;
}

//CHECK: void func3_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     float _t0 = x;
//CHECK-NEXT:     x = x - y - y * y;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     float z = y;
//CHECK-NEXT:     float _t2 = y;
//CHECK-NEXT:     float _t1 = (y = x + x);
//CHECK-NEXT:     float _d_t = 0;
//CHECK-NEXT:     float t = x * z * _t1;
//CHECK-NEXT:     _d_t += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(_d_t * t * {{.+}});
//CHECK-NEXT:         _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:         *_d_x += _d_t * _t1 * z;
//CHECK-NEXT:         _d_z += x * _d_t * _t1;
//CHECK-NEXT:         *_d_y += x * z * _d_t;
//CHECK-NEXT:         y = _t2;
//CHECK-NEXT:         float _r_d1 = *_d_y;
//CHECK-NEXT:         *_d_y = 0;
//CHECK-NEXT:         *_d_x += _r_d1;
//CHECK-NEXT:         *_d_x += _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     *_d_y += _d_z;
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         float _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x = 0;
//CHECK-NEXT:         *_d_x += _r_d0;
//CHECK-NEXT:         *_d_y += -_r_d0;
//CHECK-NEXT:         *_d_y += -_r_d0 * y;
//CHECK-NEXT:         *_d_y += y * -_r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT: }

// Function call custom derivative exists but no assign expr
float func4(float x, float y) { return std::pow(x, y); }

//CHECK: void func4_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _ret_value0 = std::pow(x, y);
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r0 = 0;
//CHECK-NEXT:         float _r1 = 0;
//CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(x, y, 1, &_r0, &_r1);
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:         *_d_y += _r1;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

// Function call custom derivative exists and is assigned
float func5(float x, float y) {
  y = std::sin(x);
  return y * y;
}

//CHECK: void func5_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     float _t0 = y;
//CHECK-NEXT:     y = std::sin(x);
//CHECK-NEXT:     _ret_value0 = y * y;
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_y += 1 * y;
//CHECK-NEXT:         *_d_y += y * 1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:         y = _t0;
//CHECK-NEXT:         float _r_d0 = *_d_y;
//CHECK-NEXT:         *_d_y = 0;
//CHECK-NEXT:         float _r0 = 0;
//CHECK-NEXT:         _r0 += _r_d0 * clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, 1.F).pushforward;
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

// Function call non custom derivative
double helper(double x, double y) { return x * y; }

//CHECK: void helper_pullback(double x, double y, double _d_y0, double *_d_x, double *_d_y, double &_final_error) {
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _ret_value0 = x * y;
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += _d_y0 * y;
//CHECK-NEXT:         *_d_y += x * _d_y0;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func6(float x, float y) {
  float z = helper(x, y);
  return z * z;
}

//CHECK: void func6_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     float z = helper(x, y);
//CHECK-NEXT:     _ret_value0 = z * z;
//CHECK-NEXT:     {
//CHECK-NEXT:         _d_z += 1 * z;
//CHECK-NEXT:         _d_z += z * 1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0;
//CHECK-NEXT:         double _r1 = 0;
//CHECK-NEXT:         double _t0 = 0;
//CHECK-NEXT:         helper_pullback(x, y, _d_z, &_r0, &_r1, _t0);
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:         *_d_y += _r1;
//CHECK-NEXT:         _final_error += _t0;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func7(float x) {
  int z = x;  // expected-warning {{Lossy assignment from 'float' to 'int'}}
  return z + z;
}

//CHECK: void func7_grad(float x, float *_d_x, double &_final_error) {
//CHECK-NEXT:     int _d_z = 0;
//CHECK-NEXT:     int z = x;
//CHECK-NEXT:     {
//CHECK-NEXT:         _d_z += 1;
//CHECK-NEXT:         _d_z += 1;
//CHECK-NEXT:     }
//CHECK-NEXT:     *_d_x += _d_z;
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT: }


double helper2(float& x) { return x * x; }

//CHECK: void helper2_pullback(float &x, double _d_y, float *_d_x, double &_final_error) {
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _ret_value0 = x * x;
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += _d_y * x;
//CHECK-NEXT:         *_d_x += x * _d_y;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func8(float x, float y) {
  float z;
  z = y + helper2(x);
  return z;
}

//CHECK: void func8_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     float z;
//CHECK-NEXT:     float _t0 = z;
//CHECK-NEXT:     float _t1 = x;
//CHECK-NEXT:     z = y + helper2(x);
//CHECK-NEXT:     _d_z += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         z = _t0;
//CHECK-NEXT:         float _r_d0 = _d_z;
//CHECK-NEXT:         _d_z = 0;
//CHECK-NEXT:         *_d_y += _r_d0;
//CHECK-NEXT:         x = _t1;
//CHECK-NEXT:         double _t2 = 0;
//CHECK-NEXT:         helper2_pullback(_t1, _r_d0, &*_d_x, _t2);
//CHECK-NEXT:         _final_error += _t2;
//CHECK-NEXT:         _final_error += std::abs(*_d_x * _t1 * {{.+}});
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT: }

float func9(float x, float y) {
  float z = helper(x, y) + helper2(x);
  z += helper2(x) * helper2(y);
  return z;
}

//CHECK: void func9_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     float _t1 = x;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     float z = helper(x, y) + helper2(x);
//CHECK-NEXT:     float _t3 = z;
//CHECK-NEXT:     float _t5 = x;
//CHECK-NEXT:     double _t7 = helper2(x);
//CHECK-NEXT:     float _t8 = y;
//CHECK-NEXT:     double _t4 = helper2(y);
//CHECK-NEXT:     z += _t7 * _t4;
//CHECK-NEXT:     _d_z += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         z = _t3;
//CHECK-NEXT:         float _r_d0 = _d_z;
//CHECK-NEXT:         x = _t5;
//CHECK-NEXT:         double _t6 = 0;
//CHECK-NEXT:         helper2_pullback(_t5, _r_d0 * _t4, &*_d_x, _t6);
//CHECK-NEXT:         y = _t8;
//CHECK-NEXT:         double _t9 = 0;
//CHECK-NEXT:         helper2_pullback(_t8, _t7 * _r_d0, &*_d_y, _t9);
//CHECK-NEXT:         _final_error += _t6 + _t9;
//CHECK-NEXT:         _final_error += std::abs(*_d_y * _t8 * {{.+}});
//CHECK-NEXT:         _final_error += std::abs(*_d_x * _t5 * {{.+}});
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0;
//CHECK-NEXT:         double _r1 = 0;
//CHECK-NEXT:         double _t0 = 0;
//CHECK-NEXT:         helper_pullback(x, y, _d_z, &_r0, &_r1, _t0);
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:         *_d_y += _r1;
//CHECK-NEXT:         x = _t1;
//CHECK-NEXT:         double _t2 = 0;
//CHECK-NEXT:         helper2_pullback(_t1, _d_z, &*_d_x, _t2);
//CHECK-NEXT:         _final_error += _t0 + _t2;
//CHECK-NEXT:         _final_error += std::abs(*_d_x * _t1 * {{.+}});
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT: }

int main() {
  clad::estimate_error(func);
  clad::estimate_error(func2);
  clad::estimate_error(func3);
  clad::estimate_error(func4);
  clad::estimate_error(func5);
  clad::estimate_error(func6);
  clad::estimate_error(func7);
  clad::estimate_error(func8);
  clad::estimate_error(func9);
}
