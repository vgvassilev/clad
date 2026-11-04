// RUN: %cladclang %s -I%S/../../include -oNestedFunctionCalls.out 2>&1 | %filecheck %s
// RUN: ./NestedFunctionCalls.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oNestedFunctionCalls.out
// RUN: ./NestedFunctionCalls.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"


double f(double x, double y) {
    return x*x + y*y;
}

double f2(double x, double y){
    double ans = f(x,y);
    return ans;
}

// CHECK: inline clad::ValueAndPushforward<double, double> f_pushforward(double x, double y, double _d_x, double _d_y) {
// CHECK-NEXT:     return {x * x + y * y, _d_x * x + x * _d_x + _d_y * y + y * _d_y};
// CHECK-NEXT: }

// CHECK: inline clad::ValueAndPushforward<double, double> f2_pushforward(double x, double y, double _d_x, double _d_y) {
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = f_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     double _d_ans = _t0.pushforward;
// CHECK-NEXT:     double ans = _t0.value;
// CHECK-NEXT:     return {ans, _d_ans};
// CHECK-NEXT: }


// CHECK: inline void f2_pushforward_pullback(double x, double y, double _d_x, double _d_y, clad::ValueAndPushforward<double, double> _d_y0, double *_d_x0, double *_d_y1);

// CHECK: void f2_hessian(double x, double y, double *hessianMatrix) {
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d_y{0., 0.};
// CHECK-NEXT:     _d_y.pushforward = 1.;
// CHECK-NEXT:     double _d_x(0.);
// CHECK-NEXT:     double _d_y0(0.);
// CHECK-NEXT:     _d_x = 1.;
// CHECK-NEXT:     f2_pushforward_pullback(x, y, _d_x, _d_y0, _d_y, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
// CHECK-NEXT:     _d_x = 0.;
// CHECK-NEXT:     _d_y0 = 1.;
// CHECK-NEXT:     f2_pushforward_pullback(x, y, _d_x, _d_y0, _d_y, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
// CHECK-NEXT:     _d_y0 = 0.;
// CHECK-NEXT: }

// CHECK: inline void f_pushforward_pullback(double x, double y, double _d_x, double _d_y, clad::ValueAndPushforward<double, double> _d_y0, double *_d_x0, double *_d_y1, double *_d_d_x, double *_d_d_y);

// CHECK: inline void f2_pushforward_pullback(double x, double y, double _d_x, double _d_y, clad::ValueAndPushforward<double, double> _d_y0, double *_d_x0, double *_d_y1) {
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d_t0 = {0., 0.};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = f_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     double _d_d_ans = 0.;
// CHECK-NEXT:     double _d_ans = _t00.pushforward;
// CHECK-NEXT:     double _d_ans0 = 0.;
// CHECK-NEXT:     double ans = _t00.value;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_ans0 += _d_y0.value;
// CHECK-NEXT:         _d_d_ans += _d_y0.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_t0.value += _d_ans0;
// CHECK-NEXT:     _d_t0.pushforward += _d_d_ans;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         double _r2 = 0.;
// CHECK-NEXT:         double _r3 = 0.;
// CHECK-NEXT:         f_pushforward_pullback(x, y, _d_x, _d_y, _d_t0, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:         *_d_x0 += _r0;
// CHECK-NEXT:         *_d_y1 += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }


// CHECK: inline void f_pushforward_pullback(double x, double y, double _d_x, double _d_y, clad::ValueAndPushforward<double, double> _d_y0, double *_d_x0, double *_d_y1, double *_d_d_x, double *_d_d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x0 += _d_y0.value * x;
// CHECK-NEXT:         *_d_x0 += x * _d_y0.value;
// CHECK-NEXT:         *_d_y1 += _d_y0.value * y;
// CHECK-NEXT:         *_d_y1 += y * _d_y0.value;
// CHECK-NEXT:         *_d_d_x += _d_y0.pushforward * x;
// CHECK-NEXT:         *_d_x0 += _d_x * _d_y0.pushforward;
// CHECK-NEXT:         *_d_x0 += _d_y0.pushforward * _d_x;
// CHECK-NEXT:         *_d_d_x += x * _d_y0.pushforward;
// CHECK-NEXT:         *_d_d_y += _d_y0.pushforward * y;
// CHECK-NEXT:         *_d_y1 += _d_y * _d_y0.pushforward;
// CHECK-NEXT:         *_d_y1 += _d_y0.pushforward * _d_y;
// CHECK-NEXT:         *_d_d_y += y * _d_y0.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// A callee that writes through an argument: the pullback of the hessian's
// pushforward restores what it overwrote before the callee's pullback runs.
void scale(double k, const double* a, double* b) { b[0] = k * a[0]; }

double chain(double k1, double k2) {
  double s[1] = {1.}, t[1];
  scale(k1, s, t);
  scale(k2, t, s);
  return s[0];
}

// Past the first element, which needs the callee's recorded stores.
void scale2(double k, const double* a, double* b) {
  b[0] = k * a[0];
  b[1] = k * a[1];
}

double looped(double k1, double k2) {
  double s[2] = {1., 1.}, t[2];
  for (int n = 0; n < 2; n++) {
    scale2(k1, s, t);
    scale2(k2, t, s);
  }
  return s[0] * s[1];
}

// One that also returns a value.
double accum(double k, double* b) {
  b[1] = b[0] * k;
  return b[1];
}

double returning(double k1, double k2) {
  double s[2] = {1., 1.};
  double r = accum(k1, s);
  r += accum(k2, s);
  return r * s[1];
}

int main() {
    auto f_hess = clad::hessian(f2);
    double mat_f[4] = {0};
    f_hess.execute(3, 4, mat_f);
    printf("[%.2f, %.2f, %.2f, %.2f]\n", mat_f[0], mat_f[1], mat_f[2], mat_f[3]); //CHECK-EXEC: [2.00, 0.00, 0.00, 2.00]
    double H[4] = {0};
    clad::hessian(chain).execute(3, 5, H);
    printf("[%.2f, %.2f, %.2f, %.2f]\n", H[0], H[1], H[2], H[3]); //CHECK-EXEC: [0.00, 1.00, 1.00, 0.00]
    double G[4] = {0};
    clad::hessian(looped).execute(2, 0.5, G);
    printf("[%.2f, %.2f, %.2f, %.2f]\n", G[0], G[1], G[2], G[3]); //CHECK-EXEC: [3.00, 16.00, 16.00, 48.00]
    double R[4] = {0};
    clad::hessian(returning).execute(2, 3, R);
    printf("[%.2f, %.2f, %.2f, %.2f]\n", R[0], R[1], R[2], R[3]); //CHECK-EXEC: [0.00, 1.00, 1.00, 2.00]
}
