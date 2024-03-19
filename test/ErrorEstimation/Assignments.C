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

//CHECK: void func_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     x = x + y;
//CHECK-NEXT:     _t1 = y;
//CHECK-NEXT:     y = x;
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     *_d_y += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         y = _t1;
//CHECK-NEXT:         float _r_d1 = *_d_y;
//CHECK-NEXT:         *_d_y -= _r_d1;
//CHECK-NEXT:         *_d_x += _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         float _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x -= _r_d0;
//CHECK-NEXT:         *_d_x += _r_d0;
//CHECK-NEXT:         *_d_y += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT: }

float func2(float x, int y) {
  x = y * x + x * x;
  return x;
}

//CHECK: void func2_grad(float x, int y, float *_d_x, int *_d_y, double &_final_error) {
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     x = y * x + x * x;
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     *_d_x += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         float _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x -= _r_d0;
//CHECK-NEXT:         *_d_y += _r_d0 * x;
//CHECK-NEXT:         *_d_x += y * _r_d0;
//CHECK-NEXT:         *_d_x += _r_d0 * x;
//CHECK-NEXT:         *_d_x += x * _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT: }

float func3(int x, int y) {
  x = y;
  return y;
}

//CHECK: void func3_grad(int x, int y, int *_d_x, int *_d_y, double &_final_error) {
//CHECK-NEXT:     int _t0;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     x = y;
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     *_d_y += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         int _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x -= _r_d0;
//CHECK-NEXT:         *_d_y += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

float func4(float x, float y) {
  double z = y;
  x = z + y;
  return x;
}

//CHECK: void func4_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     double _d_z = 0;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     double z = y;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     x = z + y;
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     *_d_x += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         float _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x -= _r_d0;
//CHECK-NEXT:         _d_z += _r_d0;
//CHECK-NEXT:         *_d_y += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     *_d_y += _d_z;
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT: }

float func5(float x, float y) {
  int z = 56;
  x = z + y;
  return x;
}

//CHECK: void func5_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     int _d_z = 0;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     int z = 56;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     x = z + y;
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     *_d_x += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         float _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x -= _r_d0;
//CHECK-NEXT:         _d_z += _r_d0;
//CHECK-NEXT:         *_d_y += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT: }

float func6(float x) { return x; }

//CHECK: void func6_grad(float x, float *_d_x, double &_final_error) {
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     *_d_x += 1;
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT: }

float func7(float x, float y) { return (x * y); }

//CHECK: void func7_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     _ret_value0 = (x * y);
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += 1 * y;
//CHECK-NEXT:         *_d_y += x * 1;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func8(int x, int y) {
    x = y * y;
    return x;
}

//CHECK: void func8_grad(int x, int y, int *_d_x, int *_d_y, double &_final_error) {
//CHECK-NEXT:     int _t0;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     x = y * y;
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     *_d_x += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         int _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x -= _r_d0;
//CHECK-NEXT:         *_d_y += _r_d0 * y;
//CHECK-NEXT:         *_d_y += y * _r_d0;
//CHECK-NEXT:     }
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
}
