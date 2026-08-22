// RUN: %cladclang %s -I%S/../../include -oTBR.out 2>&1 | %filecheck %s
// RUN: ./TBR.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oTBR.out
// RUN: ./TBR.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

double mat4[4];
clad::array_ref<double> mat4_ref(mat4, 4);

// A hessian's inner reverse pass is scheduled lazily by
// HandleNestedDiffRequest and never walked by the static planner, so it only
// runs TBR if PlanNestedRequest built it an AnalysisDeclContext. Here that
// shows up in the loop: `s = s + a` is linear, so neither s nor _d_s has to be
// taped, while `a = a * x` is not and both a and _d_a do.
double linearAndNonlinear(double x, double y) {
  double a = x * y;
  double s = a;
  for (int i = 0; i < 3; ++i) {
    s = s + a;
    a = a * x;
  }
  return s * a;
}

// CHECK: inline void linearAndNonlinear_pushforward_pullback(double x, double y, double _d_x, double _d_y, clad::ValueAndPushforward<double, double> _d_y0, double *_d_x0, double *_d_y1) {
// CHECK-NEXT:     int _d_d_i = 0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     {{__size_t|size_t|unsigned long long|unsigned long|unsigned int}} _t0;
// CHECK-NEXT:     int _d_i0 = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     double _d_d_a = 0.;
// CHECK-NEXT:     double _d_a = _d_x * y + x * _d_y;
// CHECK-NEXT:     double _d_a0 = 0.;
// CHECK-NEXT:     double a = x * y;
// CHECK-NEXT:     double _d_d_s = 0.;
// CHECK-NEXT:     double _d_s = _d_a;
// CHECK-NEXT:     double _d_s0 = 0.;
// CHECK-NEXT:     double s = a;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_i = 0;
// CHECK-NEXT:         _t0 = 0;
// CHECK-NEXT:         for (i = 0; i < 3; ++i) {
// CHECK-NEXT:             _t0++;
// CHECK-NEXT:             _d_s = _d_s + _d_a;
// CHECK-NEXT:             s = s + a;
// CHECK-NEXT:             clad::push(_t1, _d_a);
// CHECK-NEXT:             _d_a = _d_a * x + a * _d_x;
// CHECK-NEXT:             clad::push(_t2, a);
// CHECK-NEXT:             a = a * x;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_s0 += _d_y0.value * a;
// CHECK-NEXT:         _d_a0 += s * _d_y0.value;
// CHECK-NEXT:         _d_d_s += _d_y0.pushforward * a;
// CHECK-NEXT:         _d_a0 += _d_s * _d_y0.pushforward;
// CHECK-NEXT:         _d_s0 += _d_y0.pushforward * _d_a;
// CHECK-NEXT:         _d_d_a += s * _d_y0.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         for (; _t0; _t0--) {
// CHECK-NEXT:             {
// CHECK-NEXT:                 a = clad::pop(_t2);
// CHECK-NEXT:                 double _r_d3 = _d_a0;
// CHECK-NEXT:                 _d_a0 = 0.;
// CHECK-NEXT:                 _d_a0 += _r_d3 * x;
// CHECK-NEXT:                 *_d_x0 += a * _r_d3;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = clad::pop(_t1);
// CHECK-NEXT:                 double _r_d2 = _d_d_a;
// CHECK-NEXT:                 _d_d_a = 0.;
// CHECK-NEXT:                 _d_d_a += _r_d2 * x;
// CHECK-NEXT:                 *_d_x0 += _d_a * _r_d2;
// CHECK-NEXT:                 _d_a0 += _r_d2 * _d_x;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 double _r_d1 = _d_s0;
// CHECK-NEXT:                 _d_s0 = 0.;
// CHECK-NEXT:                 _d_s0 += _r_d1;
// CHECK-NEXT:                 _d_a0 += _r_d1;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 double _r_d0 = _d_d_s;
// CHECK-NEXT:                 _d_d_s = 0.;
// CHECK-NEXT:                 _d_d_s += _r_d0;
// CHECK-NEXT:                 _d_d_a += _r_d0;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_a0 += _d_s0;
// CHECK-NEXT:     _d_d_a += _d_d_s;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x0 += _d_a0 * y;
// CHECK-NEXT:         *_d_y1 += x * _d_a0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_y1 += _d_x * _d_d_a;
// CHECK-NEXT:         *_d_x0 += _d_d_a * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }
// The transcript above pins the taping exactly: two tapes for the nonlinear
// assignment, none for the linear one.

int main() {
  INIT_HESSIAN(linearAndNonlinear);

  TEST_HESSIAN(linearAndNonlinear, 1, 1.5, 2.5, mat4_ref);
  // CHECK-EXEC: {3786.33, 879.61, 879.61, 87.33}
}
