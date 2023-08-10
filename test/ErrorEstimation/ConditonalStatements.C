// RUN: %cladclang -I%S/../../include -oCondStmts.out %s 2>&1 | FileCheck %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

// Single statement if/else
float func(float x, float y) {
  if (x > y) {
    y = y * x;
  } else {
    float temp = y;
    temp = y * y;
    x = y;
  }
  return x + y;
}

//CHECK: void func_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     float _EERepl_y0 = y;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _EERepl_y1;
//CHECK-NEXT:     float _d_temp = 0;
//CHECK-NEXT:     double _delta_temp = 0;
//CHECK-NEXT:     float _EERepl_temp0;
//CHECK-NEXT:     float _t2;
//CHECK-NEXT:     float _t3;
//CHECK-NEXT:     float _EERepl_temp1;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _cond0 = x > y;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         _t1 = y;
//CHECK-NEXT:         _t0 = x;
//CHECK-NEXT:         y = _t1 * _t0;
//CHECK-NEXT:         _EERepl_y1 = y;
//CHECK-NEXT:     } else {
//CHECK-NEXT:         float temp = y;
//CHECK-NEXT:         _EERepl_temp0 = temp;
//CHECK-NEXT:         _t3 = y;
//CHECK-NEXT:         _t2 = y;
//CHECK-NEXT:         temp = _t3 * _t2;
//CHECK-NEXT:         _EERepl_temp1 = temp;
//CHECK-NEXT:         x = y;
//CHECK-NEXT:     }
//CHECK-NEXT:     _ret_value0 = x + y;
//CHECK-NEXT:     {
//CHECK-NEXT:         * _d_x += 1;
//CHECK-NEXT:         * _d_y += 1;
//CHECK-NEXT:     }
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         {
//CHECK-NEXT:             float _r_d0 = * _d_y;
//CHECK-NEXT:             float _r0 = _r_d0 * _t0;
//CHECK-NEXT:             * _d_y += _r0;
//CHECK-NEXT:             float _r1 = _t1 * _r_d0;
//CHECK-NEXT:             * _d_x += _r1;
//CHECK-NEXT:             _delta_y += std::abs(_r_d0 * _EERepl_y1 * {{.+}});
//CHECK-NEXT:             * _d_y -= _r_d0;
//CHECK-NEXT:             * _d_y;
//CHECK-NEXT:         }
//CHECK-NEXT:     } else {
//CHECK-NEXT:         {
//CHECK-NEXT:             float _r_d2 = * _d_x;
//CHECK-NEXT:             * _d_y += _r_d2;
//CHECK-NEXT:             * _d_x -= _r_d2;
//CHECK-NEXT:             * _d_x;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             float _r_d1 = _d_temp;
//CHECK-NEXT:             float _r2 = _r_d1 * _t2;
//CHECK-NEXT:             * _d_y += _r2;
//CHECK-NEXT:             float _r3 = _t3 * _r_d1;
//CHECK-NEXT:             * _d_y += _r3;
//CHECK-NEXT:             _delta_temp += std::abs(_r_d1 * _EERepl_temp1 * {{.+}});
//CHECK-NEXT:             _d_temp -= _r_d1;
//CHECK-NEXT:         }
//CHECK-NEXT:         * _d_y += _d_temp;
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * _EERepl_y0 * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y|temp}} + _delta_{{x|y|temp}} + _delta_{{x|y|temp}} + std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

// Single return statement if/else
float func2(float x) {
  float z = x * x;
  if (z > 9)
    return x + x;
  else
    return x * x;
}

//CHECK: void func2_grad(float x, clad::array_ref<float> _d_x, double &_final_error) {
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     double _delta_z = 0;
//CHECK-NEXT:     float _EERepl_z0;
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     float _t2;
//CHECK-NEXT:     float _t3;
//CHECK-NEXT:     _t1 = x;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     float z = _t1 * _t0;
//CHECK-NEXT:     _EERepl_z0 = z;
//CHECK-NEXT:     _cond0 = z > 9;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         _ret_value0 = x + x;
//CHECK-NEXT:         goto _label0;
//CHECK-NEXT:     } else {
//CHECK-NEXT:         _t3 = x;
//CHECK-NEXT:         _t2 = x;
//CHECK-NEXT:         _ret_value0 = _t3 * _t2;
//CHECK-NEXT:         goto _label1;
//CHECK-NEXT:     }
//CHECK-NEXT:     if (_cond0)
//CHECK-NEXT:       _label0:
//CHECK-NEXT:         {
//CHECK-NEXT:             * _d_x += 1;
//CHECK-NEXT:             * _d_x += 1;
//CHECK-NEXT:         }
//CHECK-NEXT:     else
//CHECK-NEXT:       _label1:
//CHECK-NEXT:         {
//CHECK-NEXT:             float _r2 = 1 * _t2;
//CHECK-NEXT:             * _d_x += _r2;
//CHECK-NEXT:             float _r3 = _t3 * 1;
//CHECK-NEXT:             * _d_x += _r3;
//CHECK-NEXT:         }
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r0 = _d_z * _t0;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:         float _r1 = _t1 * _d_z;
//CHECK-NEXT:         * _d_x += _r1;
//CHECK-NEXT:         _delta_z += std::abs(_d_z * _EERepl_z0 * {{.+}});
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|z}} + _delta_{{x|z}} + std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func3(float x, float y) { return x > 30 ? x * y : x + y; }

//CHECK: void func3_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _cond0 = x > 30;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         _t1 = x;
//CHECK-NEXT:         _t0 = y;
//CHECK-NEXT:     }
//CHECK-NEXT:     _ret_value0 = _cond0 ? _t1 * _t0 : x + y;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         float _r0 = 1 * _t0;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:         float _r1 = _t1 * 1;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:     } else {
//CHECK-NEXT:         * _d_x += 1;
//CHECK-NEXT:         * _d_y += 1;
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y}} + _delta_{{x|y}} + std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func4(float x, float y) {
  !x ? (x += 1) : (x *= x);
  return y / x;
}

//CHECK: void func4_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     float _EERepl_x0 = x;
//CHECK-NEXT:     float _EERepl_x1;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _EERepl_x2;
//CHECK-NEXT:     float _t2;
//CHECK-NEXT:     float _t3;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _cond0 = !x;
//CHECK-NEXT:     if (_cond0)
//CHECK-NEXT:         ;
//CHECK-NEXT:     else {
//CHECK-NEXT:         _t1 = x;
//CHECK-NEXT:         _t0 = x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _cond0 ? (x += 1) : (x *= _t0);
//CHECK-NEXT:     _EERepl_x2 = x;
//CHECK-NEXT:     _EERepl_x1 = x;
//CHECK-NEXT:     _t3 = y;
//CHECK-NEXT:     _t2 = x;
//CHECK-NEXT:     _ret_value0 = _t3 / _t2;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r1 = 1 / _t2;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:         float _r2 = 1 * -_t3 / (_t2 * _t2);
//CHECK-NEXT:         * _d_x += _r2;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         if (_cond0) {
//CHECK-NEXT:             float _r_d0 = * _d_x;
//CHECK-NEXT:             * _d_x += _r_d0;
//CHECK-NEXT:             _delta_x += std::abs(_r_d0 * _EERepl_x1 * {{.+}});
//CHECK-NEXT:             * _d_x -= _r_d0;
//CHECK-NEXT:         } else {
//CHECK-NEXT:             float _r_d1 = * _d_x;
//CHECK-NEXT:             * _d_x += _r_d1 * _t0;
//CHECK-NEXT:             float _r0 = _t1 * _r_d1;
//CHECK-NEXT:             * _d_x += _r0;
//CHECK-NEXT:             _delta_x += std::abs(_r_d1 * _EERepl_x2 * {{.+}});
//CHECK-NEXT:             * _d_x -= _r_d1;
//CHECK-NEXT:         }
//CHECK-NEXT:         _cond0 ? * _d_x : * _d_x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * _EERepl_x0 * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y}} + _delta_{{x|y}} + std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

int main() {
  clad::estimate_error(func);
  clad::estimate_error(func2);
  clad::estimate_error(func3);
  clad::estimate_error(func4);
}
