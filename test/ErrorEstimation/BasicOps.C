// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s
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

//CHECK: void func_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     float _EERepl_x0 = x;
//CHECK-NEXT:     float _EERepl_x1;
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     float _EERepl_y0 = y;
//CHECK-NEXT:     float _EERepl_y1;
//CHECK-NEXT:     float _EERepl_y2;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     double _delta_z = 0;
//CHECK-NEXT:     float _EERepl_z0;
//CHECK-NEXT:     x = x + y;
//CHECK-NEXT:     _EERepl_x1 = x;
//CHECK-NEXT:     _EERepl_y1 = y;
//CHECK-NEXT:     y = y + y++ + y;
//CHECK-NEXT:     _EERepl_y2 = y;
//CHECK-NEXT:     _t1 = y;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     float z = _t1 * _t0;
//CHECK-NEXT:     _EERepl_z0 = z;
//CHECK-NEXT:     _d_z += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r0 = _d_z * _t0;
//CHECK-NEXT:         * _d_y += _r0;
//CHECK-NEXT:         float _r1 = _t1 * _d_z;
//CHECK-NEXT:         * _d_x += _r1;
//CHECK-NEXT:         _delta_z += std::abs(_d_z * _EERepl_z0 * {{.+}});
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d1 = * _d_y;
//CHECK-NEXT:         * _d_y += _r_d1;
//CHECK-NEXT:         * _d_y += _r_d1;
//CHECK-NEXT:         _delta_y += std::abs(* _d_y * _EERepl_y1 * {{.+}});
//CHECK-NEXT:         * _d_y += _r_d1;
//CHECK-NEXT:         _delta_y += std::abs(_r_d1 * _EERepl_y2 * {{.+}});
//CHECK-NEXT:         * _d_y -= _r_d1;
//CHECK-NEXT:         * _d_y;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d0 = * _d_x;
//CHECK-NEXT:         * _d_x += _r_d0;
//CHECK-NEXT:         * _d_y += _r_d0;
//CHECK-NEXT:         _delta_x += std::abs(_r_d0 * _EERepl_x1 * {{.+}});
//CHECK-NEXT:         * _d_x -= _r_d0;
//CHECK-NEXT:         * _d_x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * _EERepl_x0 * {{.+}});
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * _EERepl_y0 * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y|z}} + _delta_{{x|y|z}} + _delta_{{x|y|z}};
//CHECK-NEXT: }

// This function may evaluate incorrectly due to absence of usage of
// absolute values
float func2(float x, float y) {
  x = x - y - y * y;
  float z = y / x;
  return z;
}

//CHECK: void func2_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     float _EERepl_x0 = x;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _EERepl_x1;
//CHECK-NEXT:     float _t2;
//CHECK-NEXT:     float _t3;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     double _delta_z = 0;
//CHECK-NEXT:     float _EERepl_z0;
//CHECK-NEXT:     _t1 = y;
//CHECK-NEXT:     _t0 = y;
//CHECK-NEXT:     x = x - y - _t1 * _t0;
//CHECK-NEXT:     _EERepl_x1 = x;
//CHECK-NEXT:     _t3 = y;
//CHECK-NEXT:     _t2 = x;
//CHECK-NEXT:     float z = _t3 / _t2;
//CHECK-NEXT:     _EERepl_z0 = z;
//CHECK-NEXT:     _d_z += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r2 = _d_z / _t2;
//CHECK-NEXT:         * _d_y += _r2;
//CHECK-NEXT:         float _r3 = _d_z * -_t3 / (_t2 * _t2);
//CHECK-NEXT:         * _d_x += _r3;
//CHECK-NEXT:         _delta_z += std::abs(_d_z * _EERepl_z0 * {{.+}});
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d0 = * _d_x;
//CHECK-NEXT:         * _d_x += _r_d0;
//CHECK-NEXT:         * _d_y += -_r_d0;
//CHECK-NEXT:         float _r0 = -_r_d0 * _t0;
//CHECK-NEXT:         * _d_y += _r0;
//CHECK-NEXT:         float _r1 = _t1 * -_r_d0;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:         _delta_x += std::abs(_r_d0 * _EERepl_x1 * {{.+}});
//CHECK-NEXT:         * _d_x -= _r_d0;
//CHECK-NEXT:         * _d_x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * _EERepl_x0 * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y|z}} + _delta_{{x|y|z}} + _delta_{{x|y|z}};
//CHECK-NEXT: }


// Subexpression assign
float func3(float x, float y) {
  x = x - y - y * y;
  float z = y;
  float t = x * z * (y = x + x);
  return t;
}

//CHECK: void func3_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     float _EERepl_x0 = x;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _EERepl_x1;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     double _delta_z = 0;
//CHECK-NEXT:     float _EERepl_z0;
//CHECK-NEXT:     float _t2;
//CHECK-NEXT:     float _t3;
//CHECK-NEXT:     float _t4;
//CHECK-NEXT:     float _t5;
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     float _EERepl_y0 = y;
//CHECK-NEXT:     float _EERepl_y1;
//CHECK-NEXT:     float _d_t = 0;
//CHECK-NEXT:     double _delta_t = 0;
//CHECK-NEXT:     float _EERepl_t0;
//CHECK-NEXT:     _t1 = y;
//CHECK-NEXT:     _t0 = y;
//CHECK-NEXT:     x = x - y - _t1 * _t0;
//CHECK-NEXT:     _EERepl_x1 = x;
//CHECK-NEXT:     float z = y;
//CHECK-NEXT:     _EERepl_z0 = z;
//CHECK-NEXT:     _t4 = x;
//CHECK-NEXT:     _t3 = z;
//CHECK-NEXT:     _t5 = _t4 * _t3;
//CHECK-NEXT:     _t2 = (y = x + x);
//CHECK-NEXT:     float t = _t5 * _t2;
//CHECK-NEXT:     _EERepl_t0 = t;
//CHECK-NEXT:     _EERepl_y1 = y;
//CHECK-NEXT:     _d_t += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r2 = _d_t * _t2;
//CHECK-NEXT:         float _r3 = _r2 * _t3;
//CHECK-NEXT:         * _d_x += _r3;
//CHECK-NEXT:         float _r4 = _t4 * _r2;
//CHECK-NEXT:         _d_z += _r4;
//CHECK-NEXT:         float _r5 = _t5 * _d_t;
//CHECK-NEXT:         * _d_y += _r5;
//CHECK-NEXT:         float _r_d1 = * _d_y;
//CHECK-NEXT:         * _d_x += _r_d1;
//CHECK-NEXT:         * _d_x += _r_d1;
//CHECK-NEXT:         _delta_y += std::abs(_r_d1 * _EERepl_y1 * {{.+}});
//CHECK-NEXT:         * _d_y -= _r_d1;
//CHECK-NEXT:         _delta_t += std::abs(_d_t * _EERepl_t0 * {{.+}});
//CHECK-NEXT:     }
//CHECK-NEXT:     * _d_y += _d_z;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d0 = * _d_x;
//CHECK-NEXT:         * _d_x += _r_d0;
//CHECK-NEXT:         * _d_y += -_r_d0;
//CHECK-NEXT:         float _r0 = -_r_d0 * _t0;
//CHECK-NEXT:         * _d_y += _r0;
//CHECK-NEXT:         float _r1 = _t1 * -_r_d0;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:         _delta_x += std::abs(_r_d0 * _EERepl_x1 * {{.+}});
//CHECK-NEXT:         * _d_x -= _r_d0;
//CHECK-NEXT:         * _d_x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * _EERepl_x0 * {{.+}});
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * _EERepl_y0 * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{t|x|y|z}} + _delta_{{t|x|y|z}} + _delta_{{t|x|y|z}} + _delta_{{t|x|y|z}};
//CHECK-NEXT: }

// Function call custom derivative exists but no assign expr
float func4(float x, float y) { return std::pow(x, y); }

//CHECK: void func4_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     _t1 = y;
//CHECK-NEXT:     _ret_value0 = std::pow(_t0, _t1);
//CHECK-NEXT:     {
//CHECK-NEXT:         float _grad0 = 0.F;
//CHECK-NEXT:         float _grad1 = 0.F;
//CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(_t0, _t1, 1, &_grad0, &_grad1);
//CHECK-NEXT:         float _r0 = _grad0;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:         float _r1 = _grad1;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y}} + _delta_{{x|y}} + std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

// Function call custom derivative exists and is assigned
float func5(float x, float y) {
  y = std::sin(x);
  return y * y;
}

//CHECK: void func5_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     float _EERepl_y0 = y;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _EERepl_y1;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _t2;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     y = std::sin(_t0);
//CHECK-NEXT:     _EERepl_y1 = y;
//CHECK-NEXT:     _t2 = y;
//CHECK-NEXT:     _t1 = y;
//CHECK-NEXT:     _ret_value0 = _t2 * _t1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r1 = 1 * _t1;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:         float _r2 = _t2 * 1;
//CHECK-NEXT:         * _d_y += _r2;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d0 = * _d_y;
//CHECK-NEXT:         float _r0 = _r_d0 * clad::custom_derivatives{{(::std)?}}::sin_pushforward(_t0, 1.F).pushforward;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:         _delta_y += std::abs(_r_d0 * _EERepl_y1 * {{.+}});
//CHECK-NEXT:         * _d_y -= _r_d0;
//CHECK-NEXT:         * _d_y;
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * _EERepl_y0 * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y}} + _delta_{{x|y}} + std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

// Function call non custom derivative
double helper(double x, double y) { return x * y; }

//CHECK: void helper_pullback(double x, double y, double _d_y0, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y, double &_final_error) {
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _t1 = x;
//CHECK-NEXT:     _t0 = y;
//CHECK-NEXT:     _ret_value0 = _t1 * _t0;
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = _d_y0 * _t0;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:         double _r1 = _t1 * _d_y0;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{y|x}} + _delta_{{y|x}} + std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func6(float x, float y) {
  float z = helper(x, y);
  return z * z;
}

//CHECK: void func6_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     double _delta_z = 0;
//CHECK-NEXT:     float _EERepl_z0;
//CHECK-NEXT:     float _t3;
//CHECK-NEXT:     float _t4;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     _t1 = y;
//CHECK-NEXT:     float z = helper(_t0, _t1);
//CHECK-NEXT:     _EERepl_z0 = z;
//CHECK-NEXT:     _t4 = z;
//CHECK-NEXT:     _t3 = z;
//CHECK-NEXT:     _ret_value0 = _t4 * _t3;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r2 = 1 * _t3;
//CHECK-NEXT:         _d_z += _r2;
//CHECK-NEXT:         float _r3 = _t4 * 1;
//CHECK-NEXT:         _d_z += _r3;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _grad0 = 0.;
//CHECK-NEXT:         double _grad1 = 0.;
//CHECK-NEXT:         double _t2 = 0;
//CHECK-NEXT:         helper_pullback(_t0, _t1, _d_z, &_grad0, &_grad1, _t2);
//CHECK-NEXT:         double _r0 = _grad0;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:         double _r1 = _grad1;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:         _delta_z += _t2;
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{y|x|z}} + _delta_{{y|x|z}} + _delta_{{y|x|z}} + std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func7(float x) {
  int z = x;  // expected-warning {{Lossy assignment from 'float' to 'int'}}
  return z + z;
}

//CHECK: void func7_grad(float x, clad::array_ref<float> _d_x, double &_final_error) {
//CHECK-NEXT:     int _d_z = 0;
//CHECK-NEXT:     int z = x;
//CHECK-NEXT:     {
//CHECK-NEXT:         _d_z += 1;
//CHECK-NEXT:         _d_z += 1;
//CHECK-NEXT:     }
//CHECK-NEXT:     * _d_x += _d_z;
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += _delta_x;
//CHECK-NEXT: }


double helper2(float& x) { return x * x; }

//CHECK: void helper2_pullback(float &x, double _d_y, clad::array_ref<float> _d_x, double &_final_error) {
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _t1 = x;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     _ret_value0 = _t1 * _t0;
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = _d_y * _t0;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:         double _r1 = _t1 * _d_y;
//CHECK-NEXT:         * _d_x += _r1;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func8(float x, float y) {
  float z;
  z = y + helper2(x);
  return z;
}

//CHECK: void func8_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     double _delta_z = 0;
//CHECK-NEXT:     float _EERepl_z0;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _EERepl_z1;
//CHECK-NEXT:     float z;
//CHECK-NEXT:     _EERepl_z0 = z;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     z = y + helper2(x);
//CHECK-NEXT:     _EERepl_z1 = z;
//CHECK-NEXT:     _d_z += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d0 = _d_z;
//CHECK-NEXT:         * _d_y += _r_d0;
//CHECK-NEXT:         double _t1 = 0;
//CHECK-NEXT:         helper2_pullback(_t0, _r_d0, &* _d_x, _t1);
//CHECK-NEXT:         float _r0 = * _d_x;
//CHECK-NEXT:         _delta_z += _t1;
//CHECK-NEXT:         _final_error += std::abs(_r0 * _t0 * {{.+}});
//CHECK-NEXT:         _d_z -= _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y|z}} + _delta_{{x|y|z}} + _delta_{{x|y|z}};
//CHECK-NEXT: }

float func9(float x, float y) {
  float z = helper(x, y) + helper2(x);
  z += helper2(x) * helper2(y);
  return z;
}

//CHECK: void func9_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _t3;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     double _delta_z = 0;
//CHECK-NEXT:     float _EERepl_z0;
//CHECK-NEXT:     double _t5;
//CHECK-NEXT:     float _t6;
//CHECK-NEXT:     double _t8;
//CHECK-NEXT:     float _t9;
//CHECK-NEXT:     float _EERepl_z1;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     _t1 = y;
//CHECK-NEXT:     _t3 = x;
//CHECK-NEXT:     float z = helper(_t0, _t1) + helper2(x);
//CHECK-NEXT:     _EERepl_z0 = z;
//CHECK-NEXT:     _t6 = x;
//CHECK-NEXT:     _t8 = helper2(x);
//CHECK-NEXT:     _t9 = y;
//CHECK-NEXT:     _t5 = helper2(y);
//CHECK-NEXT:     z += _t8 * _t5;
//CHECK-NEXT:     _EERepl_z1 = z;
//CHECK-NEXT:     _d_z += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d0 = _d_z;
//CHECK-NEXT:         _d_z += _r_d0;
//CHECK-NEXT:         double _r3 = _r_d0 * _t5;
//CHECK-NEXT:         double _t7 = 0;
//CHECK-NEXT:         helper2_pullback(_t6, _r3, &* _d_x, _t7);
//CHECK-NEXT:         float _r4 = * _d_x;
//CHECK-NEXT:         double _r5 = _t8 * _r_d0;
//CHECK-NEXT:         double _t10 = 0;
//CHECK-NEXT:         helper2_pullback(_t9, _r5, &* _d_y, _t10);
//CHECK-NEXT:         float _r6 = * _d_y;
//CHECK-NEXT:         _delta_z += _t7 + _t10;
//CHECK-NEXT:         _final_error += std::abs(_r6 * _t9 * {{.+}});
//CHECK-NEXT:         _final_error += std::abs(_r4 * _t6 * {{.+}});
//CHECK-NEXT:         _d_z -= _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _grad0 = 0.;
//CHECK-NEXT:         double _grad1 = 0.;
//CHECK-NEXT:         double _t2 = 0;
//CHECK-NEXT:         helper_pullback(_t0, _t1, _d_z, &_grad0, &_grad1, _t2);
//CHECK-NEXT:         double _r0 = _grad0;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:         double _r1 = _grad1;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:         double _t4 = 0;
//CHECK-NEXT:         helper2_pullback(_t3, _d_z, &* _d_x, _t4);
//CHECK-NEXT:         float _r2 = * _d_x;
//CHECK-NEXT:         _delta_z += _t2 + _t4;
//CHECK-NEXT:         _final_error += std::abs(_r2 * _t3 * {{.+}});
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y|z}} + _delta_{{x|y|z}} + _delta_{{x|y|z}};
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
