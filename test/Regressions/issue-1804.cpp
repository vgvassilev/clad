// RUN: %cladclang %s -I%S/../../include -o%t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// `obs` is a const-qualified pointer parameter, so clad cannot synthesize a
// tangent buffer for it -- its size is unknown -- and passes a null tangent
// instead, meaning "the derivative of this argument is identically zero".
// Reads through it have to yield that zero rather than dereference the null
// pointer.

double f_inner(double *params, double const *obs) {
   double arg = (obs[0] - params[1]) / params[2];
   return arg * arg;
}

double f_outer(double *params, double const *obs) {
   return f_inner(params, obs) + params[0];
}

// CHECK: clad::ValueAndPushforward<double, double> f_inner_pushforward(double *params, const double *obs, double *_d_params, const double *_d_obs) {
// CHECK-NEXT:     double _t0 = (obs[0] - params[1]);
// CHECK-NEXT:     double _d_arg = (((_d_obs ? _d_obs[0] : 0.) - _d_params[1]) * params[2] - _t0 * _d_params[2]) / (params[2] * params[2]);

// The hessian wrapper hands the pullback the same null tangent in every
// direction.
// CHECK: void f_outer_hessian_0(double *params, const double *obs, double *hessianMatrix) {
// CHECK: f_outer_pushforward_0_pullback(params, obs, _d_params, nullptr, _d_y, hessianMatrix + {{0U|0UL|0ULL}});

// The same holds for a dereference spelled without a subscript.
double g_inner(double *params, double const *obs) { return *obs * params[0]; }

double g_outer(double *params, double const *obs) { return g_inner(params, obs); }

// CHECK: clad::ValueAndPushforward<double, double> g_inner_pushforward(double *params, const double *obs, double *_d_params, const double *_d_obs) {
// CHECK-NEXT:     return {*obs * params[0], (_d_obs ? *_d_obs : 0.) * params[0] + *obs * _d_params[0]};

// Clang prints the type of a compound literal as `double [1]` up to clang 13
// and as `double[1]` from clang 14 on, so accept either spelling.
// CHECK: double g_outer_darg0_0(double *params, const double *obs) {
// CHECK-NEXT: clad::ValueAndPushforward<double, double> _t0 = g_inner_pushforward(params, obs, (double{{ ?}}[1]){1.}, nullptr);

// A local pointer inherits the null tangent of the one it is derived from. The
// classification runs over the whole function at once, so that a read that
// precedes the assignment making the tangent null -- as `p[0]` does here from
// the second iteration on -- is guarded too.
double h_inner(double *params, double const *obs) {
   const double *q = obs;
   double buf[1] = {params[0]};
   const double *p = buf;
   double s = q[0] * params[0];
   for (int i = 0; i < 2; ++i) {
      s += p[0] * params[0];
      p = obs;
   }
   return s;
}

double h_outer(double *params, double const *obs) { return h_inner(params, obs); }

// CHECK: clad::ValueAndPushforward<double, double> h_inner_pushforward(double *params, const double *obs, double *_d_params, const double *_d_obs) {
// CHECK-NEXT:     const double *_d_q = _d_obs;
// CHECK: double _d_s = (_d_q ? _d_q[0] : 0.) * params[0] + q[0] * _d_params[0];
// CHECK: _d_s += (_d_p ? _d_p[0] : 0.) * params[0] + p[0] * _d_params[0];

// The null tangent flows through the pullback chain unchanged.
// CHECK: void f_outer_pushforward_0_pullback(double *params, const double *obs, double *_d_params, const double *_d_obs, clad::ValueAndPushforward<double, double> _d_y, double *_d_params0) {
// CHECK: f_inner_pushforward_pullback(params, obs, _d_params, _d_obs, _d_t0, _d_params0);

// The pullback takes no adjoint for the tangents -- only for the requested
// parameters -- so the null tangent has no adjoint that could be written to,
// and its reads are guarded by the same hoisted condition.
// CHECK: void f_inner_pushforward_pullback(double *params, const double *obs, double *_d_params, const double *_d_obs, clad::ValueAndPushforward<double, double> _d_y, double *_d_params0) {
// CHECK: bool _cond0 = _d_obs;
// CHECK: double _d_arg = (((_cond0 ? _d_obs[0] : 0.) - _d_params[1]) * params[2] - _t00 * _d_params[2]) / _t0;

int main() {
   double params[3] = {0.5, -1, 2};
   double obs[1] = {-9.9};

   auto hess = clad::hessian(f_outer, "params[0:2]");
   double matrix[9] = {};
   hess.execute(params, obs, matrix);
   for (int i = 0; i < 3; ++i)
      printf("%.5g %.5g %.5g\n", matrix[3 * i], matrix[3 * i + 1],
             matrix[3 * i + 2]);
   // CHECK-EXEC: 0 0 0
   // CHECK-EXEC: 0 0.5 -4.45
   // CHECK-EXEC: 0 -4.45 29.704

   auto d = clad::differentiate(g_outer, "params[0]");
   printf("%.5g\n", d.execute(params, obs));
   // CHECK-EXEC: -9.9

   // s = obs0 * p0 + p0 * p0 + obs0 * p0, so ds/dp0 = 2 * obs0 + 2 * p0.
   auto d2 = clad::differentiate(h_outer, "params[0]");
   printf("%.5g\n", d2.execute(params, obs));
   // CHECK-EXEC: -18.8
}
