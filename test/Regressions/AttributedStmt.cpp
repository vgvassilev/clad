// RUN: %cladclang %s -std=c++20 -I%S/../../include -o%t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double branch(double x) {
  double r = x;
  if (x > 0) [[likely]] r *= 2;
  else [[unlikely]] r *= 3;
  return r;
}

double nested(double x) {
  double r = x;
  if (x > 0) [[likely]] {
    for (int i = 0; i < 2; ++i) [[likely]] r *= 2;
  }
  return r;
}

double labelled(double x, int i) {
  double r = 0;
  switch (i) {
    [[likely]] case 0: r = x * x; break;
    [[unlikely]] default: r = 3 * x; break;
  }
  return r;
}

int main() {
  auto df = clad::differentiate(branch, "x");
  auto gf = clad::gradient(branch);
  auto dn = clad::differentiate(nested, "x");
  auto gn = clad::gradient(nested);
  auto dl = clad::differentiate(labelled, "x");
  auto gl = clad::gradient(labelled, "x");
  for (double x : {2., -2.}) {
    double d = 0, n = 0;
    gf.execute(x, &d);
    gn.execute(x, &n);
    std::printf("%.0f %.0f %.0f %.0f\n", df.execute(x), d, dn.execute(x), n);
  }
  for (int i : {0, 1}) {
    double d = 0;
    gl.execute(2., i, &d);
    std::printf("%.0f %.0f\n", dl.execute(2., i), d);
  }
}

// CHECK-EXEC: 2 2 4 4
// CHECK-EXEC-NEXT: 3 3 1 1

// CHECK-EXEC-NEXT: 4 4
// CHECK-EXEC-NEXT: 3 3

// CHECK: double branch_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_r = _d_x;
// CHECK-NEXT:     double r = x;
// CHECK-NEXT:     if (x > 0) {
// CHECK-NEXT:         _d_r = _d_r * 2;
// CHECK-NEXT:         r *= 2;
// CHECK-NEXT:     } else {
// CHECK-NEXT:         _d_r = _d_r * 3;
// CHECK-NEXT:         r *= 3;
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK: void branch_grad(double x, double *_d_x) {
// CHECK-NEXT:     bool _cond0;
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = x;
// CHECK-NEXT:     {
// CHECK-NEXT:         _cond0 = x > 0;
// CHECK-NEXT:         if (_cond0)
// CHECK-NEXT:             r *= 2;
// CHECK-NEXT:         else
// CHECK-NEXT:             r *= 3;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     if (_cond0) {
// CHECK-NEXT:         double _r_d0 = _d_r;
// CHECK-NEXT:         _d_r = 0.;
// CHECK-NEXT:         _d_r += _r_d0 * 2;
// CHECK-NEXT:     } else {
// CHECK-NEXT:         double _r_d1 = _d_r;
// CHECK-NEXT:         _d_r = 0.;
// CHECK-NEXT:         _d_r += _r_d1 * 3;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_r;
// CHECK-NEXT: }
// CHECK: double nested_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_r = _d_x;
// CHECK-NEXT:     double r = x;
// CHECK-NEXT:     if (x > 0) {
// CHECK-NEXT:         {
// CHECK-NEXT:             int _d_i = 0;
// CHECK-NEXT:             for (int i = 0; i < 2; ++i) {
// CHECK-NEXT:                 _d_r = _d_r * 2;
// CHECK-NEXT:                 r *= 2;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK: void nested_grad(double x, double *_d_x) {
// CHECK-NEXT:     bool _cond0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = x;
// CHECK-NEXT:     {
// CHECK-NEXT:         _cond0 = x > 0;
// CHECK-NEXT:         if (_cond0) {
// CHECK-NEXT:             for (i = 0; i < 2; ++i) {
// CHECK-NEXT:                 r *= 2;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     if (_cond0)
// CHECK-NEXT:         for (_t0 = {{2U|2UL|2ULL}}; _t0; _t0--) {
// CHECK-NEXT:             double _r_d0 = _d_r;
// CHECK-NEXT:             _d_r = 0.;
// CHECK-NEXT:             _d_r += _r_d0 * 2;
// CHECK-NEXT:         }
// CHECK-NEXT:     *_d_x += _d_r;
// CHECK-NEXT: }
// CHECK: double labelled_darg0(double x, int i) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         switch (i) {
// CHECK-NEXT:           case 0:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_r = _d_x * x + x * _d_x;
// CHECK-NEXT:                 r = x * x;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           default:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_r = 3 * _d_x;
// CHECK-NEXT:                 r = 3 * x;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK: void labelled_grad_0(double x, int i, double *_d_x) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int _cond0;
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         _cond0 = i;
// CHECK-NEXT:         switch (_cond0) {
// CHECK-NEXT:             {
// CHECK-NEXT:               case 0:
// CHECK-NEXT:                 r = x * x;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:               default:
// CHECK-NEXT:                 r = 3 * x;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         switch (_cond0) {
// CHECK-NEXT:           default:
// CHECK-NEXT:             ;
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     *_d_x += 3 * _d_r;
// CHECK-NEXT:                     _d_r = 0.;
// CHECK-NEXT:                 }
// CHECK-NEXT:                 if (_cond0 != 0)
// CHECK-NEXT:                     break;
// CHECK-NEXT:             }
// CHECK-NEXT:           case 0:
// CHECK-NEXT:             ;
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     *_d_x += _d_r * x;
// CHECK-NEXT:                     *_d_x += x * _d_r;
// CHECK-NEXT:                     _d_r = 0.;
// CHECK-NEXT:                 }
// CHECK-NEXT:                 if (0 == _cond0)
// CHECK-NEXT:                     break;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }
