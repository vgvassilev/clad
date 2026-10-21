// RUN: %cladclang %s -I%S/../../include -oSimple.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./Simple.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

// The planner never sees the body of h_darg0, so its callees go unplanned.
double g(double x) { return x * x * x; }
double h(double t, double c) { return g(c + t); }
double h_darg0(double t, double c);
static auto h_darg0_clad_func = clad::differentiate(h, "t");

// CHECK: void h_darg0_grad_1(double t, double c, double *_d_c) {
// CHECK: g_pushforward_pullback(c + t, _d_c0 + _d_t0, _d_t00, &_r0, &_r1);

// The guard on the tangent of q has a null typed `const double *`.
double k(const double* p) {
  const double* q = p + 1;
  return q[0] * q[0] * p[0];
}
double m(double t, const double* c) {
  double a[2] = {c[0] + t, c[1] + t};
  return k(a);
}
double m_darg0(double t, const double* c);
static auto m_darg0_clad_func = clad::differentiate(m, "t");

float f1(float x, float y) {
  return x * x + y * y;
}

// CHECK: float f1_darg0(float x, float y) {
// CHECK-NEXT:    float _d_x = 1;
// CHECK-NEXT:    float _d_y = 0;
// CHECK-NEXT:    return _d_x * x + x * _d_x + _d_y * y + y * _d_y;
// CHECK-NEXT: }

//CHECK:   float f1_darg0_darg1(float x, float y) {
//CHECK-NEXT:       float _d_x = 0;
//CHECK-NEXT:       float _d_y = 1;
//CHECK-NEXT:       float _d__d_x = 0;
//CHECK-NEXT:       float _d_x0 = 1;
//CHECK-NEXT:       float _d__d_y = 0;
//CHECK-NEXT:       float _d_y0 = 0;
//CHECK-NEXT:       return _d__d_x * x + _d_x0 * _d_x + _d_x * _d_x0 + x * _d__d_x + _d__d_y * y + _d_y0 * _d_y + _d_y * _d_y0 + y * _d__d_y;
//CHECK-NEXT:   }

float f1_darg0(float x, float y);

int main () { // expected-no-diagnostics
  double dc = 0;
  clad::gradient(h_darg0, "c").execute(0., 2., &dc);
  printf("Result is = %f\n", dc); // CHECK-EXEC: Result is = 12.000000
  auto h_darg0_darg1 = clad::differentiate(h_darg0, "c");
  printf("Result is = %f\n", h_darg0_darg1.execute(0., 2.)); // CHECK-EXEC: Result is = 12.000000

  double c[2] = {2., 3.}, dcs[2] = {0., 0.};
  clad::gradient(m_darg0, "c").execute(0., c, dcs);
  printf("Result is = %f %f\n", dcs[0], dcs[1]); // CHECK-EXEC: Result is = 6.000000 10.000000

  auto f1_darg0_clad_func = clad::differentiate(f1, 0);
  printf("Result is = %f\n", f1_darg0_clad_func.execute(1.1, 2.1)); // CHECK-EXEC: Result is = 2.2

  auto f1_darg0_darg1_clad_func = clad::differentiate(f1_darg0, 1);
  printf("Result is = %f\n", f1_darg0_darg1_clad_func.execute(1.1, 2.1)); // CHECK-EXEC: Result is = 0

  return 0;
}

// The pullback of k is emitted after everything above.
// CHECK: void k_pushforward_pullback(const double *p, const double *_d_p, clad::ValueAndPushforward<double, double> _d_y, double *_d_p0, double *_d_d_p) {
// CHECK: double *_d_d_q = (_cond0 ? _d_d_p + 1 : nullptr);
