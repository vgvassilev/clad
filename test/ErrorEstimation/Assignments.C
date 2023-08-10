// RUN: %cladclang -I%S/../../include -oAssignments.out %s 2>&1 | FileCheck %s
// RUN: ./Assignments.out
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

float func(float x, float y) {
  x = x + y;
  y = x;
  return y;
}

//CHECK: void func_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     float _EERepl_x0 = x;
//CHECK-NEXT:     float _EERepl_x1;
//CHECK-NEXT:     x = x + y;
//CHECK-NEXT:     _EERepl_x1 = x;
//CHECK-NEXT:     y = x;
//CHECK-NEXT:     * _d_y += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d1 = * _d_y;
//CHECK-NEXT:         * _d_x += _r_d1;
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
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{y|x}} + _delta_{{y|x}};
//CHECK-NEXT: }

float func2(float x, int y) {
  x = y * x + x * x;
  return x;
}

//CHECK: void func2_grad(float x, int y, clad::array_ref<float> _d_x, clad::array_ref<int> _d_y, double &_final_error) {
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     float _EERepl_x0 = x;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     int _t1;
//CHECK-NEXT:     float _t2;
//CHECK-NEXT:     float _t3;
//CHECK-NEXT:     float _EERepl_x1;
//CHECK-NEXT:     _t1 = y;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     _t3 = x;
//CHECK-NEXT:     _t2 = x;
//CHECK-NEXT:     x = _t1 * _t0 + _t3 * _t2;
//CHECK-NEXT:     _EERepl_x1 = x;
//CHECK-NEXT:     * _d_x += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d0 = * _d_x;
//CHECK-NEXT:         float _r0 = _r_d0 * _t0;
//CHECK-NEXT:         * _d_y += _r0;
//CHECK-NEXT:         float _r1 = _t1 * _r_d0;
//CHECK-NEXT:         * _d_x += _r1;
//CHECK-NEXT:         float _r2 = _r_d0 * _t2;
//CHECK-NEXT:         * _d_x += _r2;
//CHECK-NEXT:         float _r3 = _t3 * _r_d0;
//CHECK-NEXT:         * _d_x += _r3;
//CHECK-NEXT:         _delta_x += std::abs(_r_d0 * _EERepl_x1 * {{.+}});
//CHECK-NEXT:         * _d_x -= _r_d0;
//CHECK-NEXT:         * _d_x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * _EERepl_x0 * {{.+}});
//CHECK-NEXT:     _final_error += _delta_x;
//CHECK-NEXT: }

float func3(int x, int y) {
  x = y;
  return y;
}

//CHECK: void func3_grad(int x, int y, clad::array_ref<int> _d_x, clad::array_ref<int> _d_y, double &_final_error) {
//CHECK-NEXT:     x = y;
//CHECK-NEXT:     * _d_y += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         int _r_d0 = * _d_x;
//CHECK-NEXT:         * _d_y += _r_d0;
//CHECK-NEXT:         * _d_x -= _r_d0;
//CHECK-NEXT:         * _d_x;
//CHECK-NEXT:     }
//CHECK-NEXT: }

float func4(float x, float y) {
  double z = y;
  x = z + y;
  return x;
}

//CHECK: void func4_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     double _d_z = 0;
//CHECK-NEXT:     double _delta_z = 0;
//CHECK-NEXT:     double _EERepl_z0;
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     float _EERepl_x0 = x;
//CHECK-NEXT:     float _EERepl_x1;
//CHECK-NEXT:     double z = y;
//CHECK-NEXT:     _EERepl_z0 = z;
//CHECK-NEXT:     x = z + y;
//CHECK-NEXT:     _EERepl_x1 = x;
//CHECK-NEXT:     * _d_x += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d0 = * _d_x;
//CHECK-NEXT:         _d_z += _r_d0;
//CHECK-NEXT:         * _d_y += _r_d0;
//CHECK-NEXT:         _delta_x += std::abs(_r_d0 * _EERepl_x1 * {{.+}});
//CHECK-NEXT:         * _d_x -= _r_d0;
//CHECK-NEXT:         * _d_x;
//CHECK-NEXT:     }
//CHECK-NEXT:     * _d_y += _d_z;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * _EERepl_x0 * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y|z}} + _delta_{{x|y|z}} + _delta_{{x|y|z}};
//CHECK-NEXT: }

float func5(float x, float y) {
  int z = 56;
  x = z + y;
  return x;
}

//CHECK: void func5_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     int _d_z = 0;
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     float _EERepl_x0 = x;
//CHECK-NEXT:     float _EERepl_x1;
//CHECK-NEXT:     int z = 56;
//CHECK-NEXT:     x = z + y;
//CHECK-NEXT:     _EERepl_x1 = x;
//CHECK-NEXT:     * _d_x += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d0 = * _d_x;
//CHECK-NEXT:         _d_z += _r_d0;
//CHECK-NEXT:         * _d_y += _r_d0;
//CHECK-NEXT:         _delta_x += std::abs(_r_d0 * _EERepl_x1 * {{.+}});
//CHECK-NEXT:         * _d_x -= _r_d0;
//CHECK-NEXT:         * _d_x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * _EERepl_x0 * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y}} + _delta_{{x|y}};
//CHECK-NEXT: }

float func6(float x) { return x; }

//CHECK: void func6_grad(float x, clad::array_ref<float> _d_x, double &_final_error) {
//CHECK-NEXT:     * _d_x += 1;
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += _delta_x;
//CHECK-NEXT: }

float func7(float x, float y) { return (x * y); }

//CHECK: void func7_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _t1 = x;
//CHECK-NEXT:     _t0 = y;
//CHECK-NEXT:     _ret_value0 = (_t1 * _t0);
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r0 = 1 * _t0;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:         float _r1 = _t1 * 1;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:     }
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += _delta_{{x|y}} + _delta_{{x|y}} + std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

int main() {
  clad::estimate_error(func);
  clad::estimate_error(func2);
  clad::estimate_error(func3);
  clad::estimate_error(func4);
  clad::estimate_error(func5);
  clad::estimate_error(func6);
  clad::estimate_error(func7);
}
