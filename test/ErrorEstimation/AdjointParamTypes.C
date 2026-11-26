// RUN: %cladclang -I%S/../../include -oAdjointParamTypes.out %s 2>&1 | %filecheck %s
// RUN: ./AdjointParamTypes.out | %filecheck_exec %s

// clad::estimate_error calls the derivative it generates through a pointer
// whose type GradientDerivedEstFnTraits spells out. That type has to be the
// derivative's own: calling a function through a pointer to another function
// type is undefined behaviour, and no test would notice. The generated
// signatures are checked below, and the static_asserts check the trait against
// the same signatures.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>
#include <type_traits>

double f_val(double x, double s) { return x * s; }
double f_ptr(double* p, double s) { return p[0] * s; }
double f_cptr(const double* p, double s) { return p[0] * s; }
double f_ref(double& x, double s) { return x * s; }
double f_cref(const double& x, double s) { return x * s; }
double f_arr(double (&a)[3], double s) { return a[0] * s; }
double f_carr(const double (&a)[3], double s) { return a[0] * s; }
double f_cptrc(const double* const p, double s) { return p[0] * s; }
double f_pptr(double** p, double s) { return p[0][0] * s; }
double f_pref(double*& p, double s) { return p[0] * s; }
double f_vptr(volatile double* p, double s) { return p[0] * s; }
double f_vref(volatile double& x, double s) { return x * s; }
float f_float(float x, float s) { return x * s; }
double f_int(double x, int n) { return x * n; }

template <class F> using Est = clad::GradientDerivedEstFnTraits_t<F>;

static_assert(std::is_same<Est<decltype(&f_val)>,
                           void (*)(double, double, double*, double*,
                                    double&)>::value, "double");
static_assert(std::is_same<Est<decltype(&f_ptr)>,
                           void (*)(double*, double, double*, double*,
                                    double&)>::value, "double*");
static_assert(std::is_same<Est<decltype(&f_cptr)>,
                           void (*)(const double*, double, double*, double*,
                                    double&)>::value, "const double*");
static_assert(std::is_same<Est<decltype(&f_ref)>,
                           void (*)(double&, double, double*, double*,
                                    double&)>::value, "double&");
static_assert(std::is_same<Est<decltype(&f_cref)>,
                           void (*)(const double&, double, double*, double*,
                                    double&)>::value, "const double&");
static_assert(std::is_same<Est<decltype(&f_arr)>,
                           void (*)(double (&)[3], double, double (*)[3],
                                    double*, double&)>::value, "double(&)[3]");
static_assert(std::is_same<Est<decltype(&f_carr)>,
                           void (*)(const double (&)[3], double,
                                    double (*)[3], double*,
                                    double&)>::value, "const double(&)[3]");
static_assert(std::is_same<Est<decltype(&f_cptrc)>,
                           void (*)(const double* const, double, double*,
                                    double*, double&)>::value,
              "const double* const");
static_assert(std::is_same<Est<decltype(&f_pptr)>,
                           void (*)(double**, double, double**, double*,
                                    double&)>::value, "double**");
// A reference to a pointer keeps its pointer level, and only const is dropped
// from the adjoint, not volatile.
static_assert(std::is_same<Est<decltype(&f_pref)>,
                           void (*)(double*&, double, double**, double*,
                                    double&)>::value, "double*&");
static_assert(std::is_same<Est<decltype(&f_vptr)>,
                           void (*)(volatile double*, double, volatile double*,
                                    double*, double&)>::value,
              "volatile double*");
static_assert(std::is_same<Est<decltype(&f_vref)>,
                           void (*)(volatile double&, double, volatile double*,
                                    double*, double&)>::value,
              "volatile double&");
static_assert(std::is_same<Est<decltype(&f_float)>,
                           void (*)(float, float, float*, float*,
                                    double&)>::value, "float");
static_assert(std::is_same<Est<decltype(&f_int)>,
                           void (*)(double, int, double*, int*,
                                    double&)>::value, "int");

// The gradient and Jacobian traits do not know which adjoints a call asks for,
// so they keep void* for every parameter.
static_assert(std::is_same<clad::OutputParamType_t<const double&, void>,
                           void*>::value, "void*");
static_assert(std::is_same<clad::GradientDerivedFnTraits_t<decltype(&f_cref)>,
                           void (*)(const double&, double, void*,
                                    void*)>::value, "gradient");

//CHECK: void f_val_grad(double x, double s, double *_d_x, double *_d_s, double &_final_error) {
//CHECK: void f_ptr_grad(double *p, double s, double *_d_p, double *_d_s, double &_final_error) {
//CHECK: void f_cptr_grad(const double *p, double s, double *_d_p, double *_d_s, double &_final_error) {
//CHECK: void f_ref_grad(double &x, double s, double *_d_x, double *_d_s, double &_final_error) {
//CHECK: void f_cref_grad(const double &x, double s, double *_d_x, double *_d_s, double &_final_error) {
//CHECK: void f_arr_grad(double (&a)[3], double s, double (*_d_a)[3], double *_d_s, double &_final_error) {
//CHECK: void f_carr_grad(const double (&a)[3], double s, double (*_d_a)[3], double *_d_s, double &_final_error) {
//CHECK: void f_pref_grad(double *&p, double s, double **_d_p, double *_d_s, double &_final_error) {
//CHECK: void f_vptr_grad(volatile double *p, double s, volatile double *_d_p, double *_d_s, double &_final_error) {
//CHECK: void f_vref_grad(volatile double &x, double s, volatile double *_d_x, double *_d_s, double &_final_error) {
//CHECK: void f_cptrc_grad(const double *const p, double s, double *_d_p, double *_d_s, double &_final_error) {
//CHECK: void f_pptr_grad(double **p, double s, double **_d_p, double *_d_s, double &_final_error) {
//CHECK: void f_float_grad(float x, float s, float *_d_x, float *_d_s, double &_final_error) {
//CHECK: void f_int_grad(double x, int n, double *_d_x, int *_d_n, double &_final_error) {

int main() {
  double x = 2, s = 3, dx = 0, ds = 0, err = 0;
  double p[1] = {2}, dp[1] = {0};
  double a[3] = {2, 0, 0}, da[3] = {0, 0, 0};

  auto e_val = clad::estimate_error(f_val);
  e_val.execute(x, s, &dx, &ds, err);
  printf("val: %g %g\n", dx, ds); // CHECK-EXEC: val: 3 2

  dp[0] = ds = 0;
  auto e_ptr = clad::estimate_error(f_ptr);
  e_ptr.execute(p, s, dp, &ds, err);
  printf("ptr: %g %g\n", dp[0], ds); // CHECK-EXEC: ptr: 3 2

  dp[0] = ds = 0;
  auto e_cptr = clad::estimate_error(f_cptr);
  e_cptr.execute(p, s, dp, &ds, err);
  printf("cptr: %g %g\n", dp[0], ds); // CHECK-EXEC: cptr: 3 2

  dx = ds = 0;
  auto e_ref = clad::estimate_error(f_ref);
  e_ref.execute(x, s, &dx, &ds, err);
  printf("ref: %g %g\n", dx, ds); // CHECK-EXEC: ref: 3 2

  dx = ds = 0;
  auto e_cref = clad::estimate_error(f_cref);
  e_cref.execute(x, s, &dx, &ds, err);
  printf("cref: %g %g\n", dx, ds); // CHECK-EXEC: cref: 3 2

  ds = 0;
  auto e_arr = clad::estimate_error(f_arr);
  e_arr.execute(a, s, &da, &ds, err);
  printf("arr: %g %g\n", da[0], ds); // CHECK-EXEC: arr: 3 2

  da[0] = ds = 0;
  auto e_carr = clad::estimate_error(f_carr);
  e_carr.execute(a, s, &da, &ds, err);
  printf("carr: %g %g\n", da[0], ds); // CHECK-EXEC: carr: 3 2

  double* pp = p;
  double* dpp = dp;
  dp[0] = ds = 0;
  auto e_pref = clad::estimate_error(f_pref);
  e_pref.execute(pp, s, &dpp, &ds, err);
  printf("pref: %g %g\n", dp[0], ds); // CHECK-EXEC: pref: 3 2

  // Only the generated signatures and the trait are checked for these.
  auto e_vptr = clad::estimate_error(f_vptr);
  auto e_vref = clad::estimate_error(f_vref);
  auto e_cptrc = clad::estimate_error(f_cptrc);
  auto e_pptr = clad::estimate_error(f_pptr);
  auto e_float = clad::estimate_error(f_float);
  auto e_int = clad::estimate_error(f_int);
}
