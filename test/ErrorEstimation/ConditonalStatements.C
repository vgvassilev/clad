// RUN: %cladclang -I%S/../../include -oCondStmts.out %s 2>&1 | %filecheck %s
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

//CHECK: void func_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _d_temp = 0;
//CHECK-NEXT:     float temp = 0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _t2;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     {
//CHECK-NEXT:     _cond0 = x > y;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         _t0 = y;
//CHECK-NEXT:         y = y * x;
//CHECK-NEXT:     } else {
//CHECK-NEXT:         temp = y;
//CHECK-NEXT:         _t1 = temp;
//CHECK-NEXT:         temp = y * y;
//CHECK-NEXT:         _t2 = x;
//CHECK-NEXT:         x = y;
//CHECK-NEXT:     }
//CHECK-NEXT:     }
//CHECK-NEXT:     _ret_value0 = x + y;
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += 1;
//CHECK-NEXT:         *_d_y += 1;
//CHECK-NEXT:     }
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         {
//CHECK-NEXT:             _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:             y = _t0;
//CHECK-NEXT:             float _r_d0 = *_d_y;
//CHECK-NEXT:             *_d_y = 0;
//CHECK-NEXT:             *_d_y += _r_d0 * x;
//CHECK-NEXT:             *_d_x += y * _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:     } else {
//CHECK-NEXT:         {
//CHECK-NEXT:             x = _t2;
//CHECK-NEXT:             float _r_d2 = *_d_x;
//CHECK-NEXT:             *_d_x = 0;
//CHECK-NEXT:             *_d_y += _r_d2;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             _final_error += std::abs(_d_temp * temp * {{.+}});
//CHECK-NEXT:             temp = _t1;
//CHECK-NEXT:             float _r_d1 = _d_temp;
//CHECK-NEXT:             _d_temp = 0;
//CHECK-NEXT:             *_d_y += _r_d1 * y;
//CHECK-NEXT:             *_d_y += y * _r_d1;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             _final_error += std::abs(_d_temp * temp * {{.+}});
//CHECK-NEXT:             *_d_y += _d_temp;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

// Single return statement if/else
float func2(float x) {
  float z = x * x;
  if (z > 9)
    return x + x;
  else
    return x * x;
}

//CHECK: void func2_grad(float x, float *_d_x, double &_final_error) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     float _d_z = 0;
//CHECK-NEXT:     float z = x * x;
//CHECK-NEXT:     {
//CHECK-NEXT:     _cond0 = z > 9;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         _ret_value0 = x + x;
//CHECK-NEXT:         goto _label0;
//CHECK-NEXT:     } else {
//CHECK-NEXT:         _ret_value0 = x * x;
//CHECK-NEXT:         goto _label1;
//CHECK-NEXT:     }
//CHECK-NEXT:     }
//CHECK-NEXT:     if (_cond0)
//CHECK-NEXT:       _label0:
//CHECK-NEXT:         {
//CHECK-NEXT:             *_d_x += 1;
//CHECK-NEXT:             *_d_x += 1;
//CHECK-NEXT:         }
//CHECK-NEXT:     else
//CHECK-NEXT:       _label1:
//CHECK-NEXT:         {
//CHECK-NEXT:             *_d_x += 1 * x;
//CHECK-NEXT:             *_d_x += x * 1;
//CHECK-NEXT:         }
//CHECK-NEXT:     {
//CHECK-NEXT:         _final_error += std::abs(_d_z * z * {{.+}});
//CHECK-NEXT:         *_d_x += _d_z * x;
//CHECK-NEXT:         *_d_x += x * _d_z;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func3(float x, float y) { return x > 30 ? x * y : x + y; }

//CHECK: void func3_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     bool _cond0 = x > 30;
//CHECK-NEXT:     _ret_value0 = _cond0 ? x * y : x + y;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         *_d_x += 1 * y;
//CHECK-NEXT:         *_d_y += x * 1;
//CHECK-NEXT:     } else {
//CHECK-NEXT:         *_d_x += 1;
//CHECK-NEXT:         *_d_y += 1;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

float func4(float x, float y) {
  !x ? (x += 1) : (x *= x);
  return y / x;
}

//CHECK: void func4_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     double _ret_value0 = 0;
//CHECK-NEXT:     bool _cond0 = !x;
//CHECK-NEXT:     if (_cond0)
//CHECK-NEXT:         _t0 = x;
//CHECK-NEXT:     else
//CHECK-NEXT:         _t1 = x;
//CHECK-NEXT:     _cond0 ? (x += 1) : (x *= x);
//CHECK-NEXT:     _ret_value0 = y / x;
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_y += 1 / x;
//CHECK-NEXT:         float _r0 = 1 * -(y / (x * x));
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         float _r_d0 = *_d_x;
//CHECK-NEXT:     } else {
//CHECK-NEXT:         _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:         x = _t1;
//CHECK-NEXT:         float _r_d1 = *_d_x;
//CHECK-NEXT:         *_d_x = 0;
//CHECK-NEXT:         *_d_x += _r_d1 * x;
//CHECK-NEXT:         *_d_x += x * _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     _final_error += std::abs(*_d_x * x * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(*_d_y * y * {{.+}});
//CHECK-NEXT:     _final_error += std::abs(1. * _ret_value0 * {{.+}});
//CHECK-NEXT: }

int main() {
  clad::estimate_error(func);
  clad::estimate_error(func2);
  clad::estimate_error(func3);
  clad::estimate_error(func4);
}
