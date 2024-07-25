// RUN: %cladclang %s -I%S/../../include -oNestedFunctionCalls.out 2>&1 | %filecheck %s
// RUN: ./NestedFunctionCalls.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oNestedFunctionCalls.out
// RUN: ./NestedFunctionCalls.out | %filecheck_exec %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"


double f(double x, double y) {
    return x*x + y*y;
}

double f2(double x, double y){
    double ans = f(x,y);
    return ans;
}

// CHECK: double f2_darg0(double x, double y);
// CHECK: void f2_darg0_grad(double x, double y, double *_d_x, double *_d_y);
// CHECK: double f2_darg1(double x, double y);
// CHECK: void f2_darg1_grad(double x, double y, double *_d_x, double *_d_y);

// CHECK: void f2_hessian(double x, double y, double *hessianMatrix) {
// CHECK-NEXT:     f2_darg0_grad(x, y, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
// CHECK-NEXT:     f2_darg1_grad(x, y, hessianMatrix + {{2U|2UL}}, hessianMatrix + {{3U|3UL}});
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<double, double> f_pushforward(double x, double y, double _d_x, double _d_y);

// CHECK: double f2_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = f_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     double _d_ans = _t0.pushforward;
// CHECK-NEXT:     double ans = _t0.value;
// CHECK-NEXT:     return _d_ans;
// CHECK-NEXT: }

// CHECK: void f_pushforward_pullback(double x, double y, double _d_x, double _d_y, clad::ValueAndPushforward<double, double> _d_y0, double *_d_x, double *_d_y, double *_d__d_x, double *_d__d_y);

// CHECK: void f2_darg0_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d_x0 = 1;
// CHECK-NEXT:     double _d__d_y = 0;
// CHECK-NEXT:     double _d_y0 = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d__t0 = {};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = f_pushforward(x, y, _d_x0, _d_y0);
// CHECK-NEXT:     double _d__d_ans = 0;
// CHECK-NEXT:     double _d_ans = _t00.pushforward;
// CHECK-NEXT:     double _d_ans0 = 0;
// CHECK-NEXT:     double ans = _t00.value;
// CHECK-NEXT:     _d__d_ans += 1;
// CHECK-NEXT:     _d__t0.value += _d_ans0;
// CHECK-NEXT:     _d__t0.pushforward += _d__d_ans;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         double _r1 = 0;
// CHECK-NEXT:         double _r2 = 0;
// CHECK-NEXT:         double _r3 = 0;
// CHECK-NEXT:         f_pushforward_pullback(x, y, _d_x0, _d_y0, _d__t0, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:         _d__d_x += _r2;
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

// CHECK: void f2_darg1_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d_x0 = 0;
// CHECK-NEXT:     double _d__d_y = 0;
// CHECK-NEXT:     double _d_y0 = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d__t0 = {};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = f_pushforward(x, y, _d_x0, _d_y0);
// CHECK-NEXT:     double _d__d_ans = 0;
// CHECK-NEXT:     double _d_ans = _t00.pushforward;
// CHECK-NEXT:     double _d_ans0 = 0;
// CHECK-NEXT:     double ans = _t00.value;
// CHECK-NEXT:     _d__d_ans += 1;
// CHECK-NEXT:     _d__t0.value += _d_ans0;
// CHECK-NEXT:     _d__t0.pushforward += _d__d_ans;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         double _r1 = 0;
// CHECK-NEXT:         double _r2 = 0;
// CHECK-NEXT:         double _r3 = 0;
// CHECK-NEXT:         f_pushforward_pullback(x, y, _d_x0, _d_y0, _d__t0, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:         _d__d_y += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<double, double> f_pushforward(double x, double y, double _d_x, double _d_y) {
// CHECK-NEXT:     return {x * x + y * y, _d_x * x + x * _d_x + _d_y * y + y * _d_y};
// CHECK-NEXT: }

// CHECK: void f_pushforward_pullback(double x, double y, double _d_x, double _d_y, clad::ValueAndPushforward<double, double> _d_y0, double *_d_x, double *_d_y, double *_d__d_x, double *_d__d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += _d_y0.value * x;
// CHECK-NEXT:         *_d_x += x * _d_y0.value;
// CHECK-NEXT:         *_d_y += _d_y0.value * y;
// CHECK-NEXT:         *_d_y += y * _d_y0.value;
// CHECK-NEXT:         *_d__d_x += _d_y0.pushforward * x;
// CHECK-NEXT:         *_d_x += _d_x * _d_y0.pushforward;
// CHECK-NEXT:         *_d_x += _d_y0.pushforward * _d_x;
// CHECK-NEXT:         *_d__d_x += x * _d_y0.pushforward;
// CHECK-NEXT:         *_d__d_y += _d_y0.pushforward * y;
// CHECK-NEXT:         *_d_y += _d_y * _d_y0.pushforward;
// CHECK-NEXT:         *_d_y += _d_y0.pushforward * _d_y;
// CHECK-NEXT:         *_d__d_y += y * _d_y0.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
    auto f_hess = clad::hessian(f2);
    double mat_f[4] = {0};
    f_hess.execute(3, 4, mat_f);
    printf("[%.2f, %.2f, %.2f, %.2f]\n", mat_f[0], mat_f[1], mat_f[2], mat_f[3]); //CHECK-EXEC: [2.00, 0.00, 0.00, 2.00]
}
