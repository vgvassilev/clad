// RUN: %cladclang %s -I%S/../../include -oNestedFunctionCalls.out 2>&1 | FileCheck %s
// RUN: ./NestedFunctionCalls.out | FileCheck -check-prefix=CHECK-EXEC %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"


double f(double x, double y) {
    return x*x + y*y;
}

double f2(double x, double y){
    double ans = f(x,y);
    return ans;
}

// CHECK: clad::ValueAndPushforward<double, double> f_pushforward(double x, double y, double _d_x, double _d_y) {
// CHECK-NEXT:     return {x * x + y * y, _d_x * x + x * _d_x + _d_y * y + y * _d_y};
// CHECK-NEXT: }

// CHECK: double f2_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = f_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     double _d_ans = _t0.pushforward;
// CHECK-NEXT:     double ans = _t0.value;
// CHECK-NEXT:     return _d_ans;
// CHECK-NEXT: }

// CHECK: void f_pushforward_pullback(double x, double y, double _d_x, double _d_y, clad::ValueAndPushforward<double, double> _d_y0, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y, clad::array_ref<double> _d__d_x, clad::array_ref<double> _d__d_y) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     double _t6;
// CHECK-NEXT:     double _t7;
// CHECK-NEXT:     double _t8;
// CHECK-NEXT:     double _t9;
// CHECK-NEXT:     double _t10;
// CHECK-NEXT:     double _t11;
// CHECK-NEXT:     _t1 = x;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t3 = y;
// CHECK-NEXT:     _t2 = y;
// CHECK-NEXT:     _t5 = _d_x;
// CHECK-NEXT:     _t4 = x;
// CHECK-NEXT:     _t7 = x;
// CHECK-NEXT:     _t6 = _d_x;
// CHECK-NEXT:     _t9 = _d_y;
// CHECK-NEXT:     _t8 = y;
// CHECK-NEXT:     _t11 = y;
// CHECK-NEXT:     _t10 = _d_y;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = _d_y0.value * _t0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         double _r1 = _t1 * _d_y0.value;
// CHECK-NEXT:         * _d_x += _r1;
// CHECK-NEXT:         double _r2 = _d_y0.value * _t2;
// CHECK-NEXT:         * _d_y += _r2;
// CHECK-NEXT:         double _r3 = _t3 * _d_y0.value;
// CHECK-NEXT:         * _d_y += _r3;
// CHECK-NEXT:         double _r4 = _d_y0.pushforward * _t4;
// CHECK-NEXT:         * _d__d_x += _r4;
// CHECK-NEXT:         double _r5 = _t5 * _d_y0.pushforward;
// CHECK-NEXT:         * _d_x += _r5;
// CHECK-NEXT:         double _r6 = _d_y0.pushforward * _t6;
// CHECK-NEXT:         * _d_x += _r6;
// CHECK-NEXT:         double _r7 = _t7 * _d_y0.pushforward;
// CHECK-NEXT:         * _d__d_x += _r7;
// CHECK-NEXT:         double _r8 = _d_y0.pushforward * _t8;
// CHECK-NEXT:         * _d__d_y += _r8;
// CHECK-NEXT:         double _r9 = _t9 * _d_y0.pushforward;
// CHECK-NEXT:         * _d_y += _r9;
// CHECK-NEXT:         double _r10 = _d_y0.pushforward * _t10;
// CHECK-NEXT:         * _d_y += _r10;
// CHECK-NEXT:         double _r11 = _t11 * _d_y0.pushforward;
// CHECK-NEXT:         * _d__d_y += _r11;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f2_darg0_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d__d_y = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d__t0 = {};
// CHECK-NEXT:     double _d__d_ans = 0;
// CHECK-NEXT:     double _d_ans0 = 0;
// CHECK-NEXT:     double _d_x0 = 1;
// CHECK-NEXT:     double _d_y0 = 0;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = y;
// CHECK-NEXT:     _t2 = _d_x0;
// CHECK-NEXT:     _t3 = _d_y0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = f_pushforward(_t0, _t1, _t2, _t3);
// CHECK-NEXT:     double _d_ans = _t00.pushforward;
// CHECK-NEXT:     double ans = _t00.value;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d__d_ans += 1;
// CHECK-NEXT:     _d__t0.value += _d_ans0;
// CHECK-NEXT:     _d__t0.pushforward += _d__d_ans;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _grad0 = 0.;
// CHECK-NEXT:         double _grad1 = 0.;
// CHECK-NEXT:         double _grad2 = 0.;
// CHECK-NEXT:         double _grad3 = 0.;
// CHECK-NEXT:         f_pushforward_pullback(_t0, _t1, _t2, _t3, _d__t0, &_grad0, &_grad1, &_grad2, &_grad3);
// CHECK-NEXT:         double _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         double _r1 = _grad1;
// CHECK-NEXT:         * _d_y += _r1;
// CHECK-NEXT:         double _r2 = _grad2;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:         double _r3 = _grad3;
// CHECK-NEXT:         _d__d_y += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: double f2_darg1(double x, double y) {
// CHECK-NEXT:     double _d_x = 0;
// CHECK-NEXT:     double _d_y = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = f_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     double _d_ans = _t0.pushforward;
// CHECK-NEXT:     double ans = _t0.value;
// CHECK-NEXT:     return _d_ans;
// CHECK-NEXT: }

// CHECK: void f2_darg1_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d__d_y = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d__t0 = {};
// CHECK-NEXT:     double _d__d_ans = 0;
// CHECK-NEXT:     double _d_ans0 = 0;
// CHECK-NEXT:     double _d_x0 = 0;
// CHECK-NEXT:     double _d_y0 = 1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = y;
// CHECK-NEXT:     _t2 = _d_x0;
// CHECK-NEXT:     _t3 = _d_y0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = f_pushforward(_t0, _t1, _t2, _t3);
// CHECK-NEXT:     double _d_ans = _t00.pushforward;
// CHECK-NEXT:     double ans = _t00.value;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d__d_ans += 1;
// CHECK-NEXT:     _d__t0.value += _d_ans0;
// CHECK-NEXT:     _d__t0.pushforward += _d__d_ans;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _grad0 = 0.;
// CHECK-NEXT:         double _grad1 = 0.;
// CHECK-NEXT:         double _grad2 = 0.;
// CHECK-NEXT:         double _grad3 = 0.;
// CHECK-NEXT:         f_pushforward_pullback(_t0, _t1, _t2, _t3, _d__t0, &_grad0, &_grad1, &_grad2, &_grad3);
// CHECK-NEXT:         double _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         double _r1 = _grad1;
// CHECK-NEXT:         * _d_y += _r1;
// CHECK-NEXT:         double _r2 = _grad2;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:         double _r3 = _grad3;
// CHECK-NEXT:         _d__d_y += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f2_hessian(double x, double y, clad::array_ref<double> hessianMatrix) {
// CHECK-NEXT:     f2_darg0_grad(x, y, hessianMatrix.slice(0UL, 1UL), hessianMatrix.slice(1UL, 1UL));
// CHECK-NEXT:     f2_darg1_grad(x, y, hessianMatrix.slice(2UL, 1UL), hessianMatrix.slice(3UL, 1UL));
// CHECK-NEXT: }

int main() {
    auto f_hess = clad::hessian(f2);
    double mat_f[4] = {0};
    clad::array_ref<double> mat_f_ref(mat_f, 4);
    f_hess.execute(3, 4, mat_f_ref);
    printf("[%.2f, %.2f, %.2f, %.2f]\n", mat_f_ref[0], mat_f_ref[1], mat_f_ref[2], mat_f_ref[3]); //CHECK-EXEC: [2.00, 0.00, 0.00, 2.00]
}
