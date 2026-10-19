// RUN: %cladclang %s -I%S/../../include -oConcurrency.out 2>&1 | %filecheck %s
// RUN: ./Concurrency.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <cstdio>
#include <functional>

// --- helpers ---

double read_ref(std::reference_wrapper<double> r) { return r.get() * r.get(); }

double read_cref(std::reference_wrapper<const double> r) {
  return r.get() * r.get();
}

double read_implicit(std::reference_wrapper<double> r) {
  double& v = r;
  return v * v;
}

double mul_refs(std::reference_wrapper<double> a,
                std::reference_wrapper<double> b) {
  return a.get() * b.get();
}

double read_cref_implicit(std::reference_wrapper<const double> r) {
  const double& v = r;
  return v * v;
}

struct Holder {
  double a;
};

// --- reverse-mode functions ---

double f_ref(double x) { return read_ref(std::ref(x)); }

double f_cref(double x) { return read_cref(std::cref(x)); }

double f_implicit(double x) { return read_implicit(std::ref(x)); }

double f_twice_ref(double x) {
  return read_ref(std::ref(x)) + read_ref(std::ref(x));
}

double f_sum_refs(double x, double y) {
  return read_ref(std::ref(x)) + read_ref(std::ref(y));
}

double f_mul_refs(double x, double y) {
  return mul_refs(std::ref(x), std::ref(y));
}

// A wrapper held in a local variable, rather than passed straight to a callee.
double f_local_ref(double x) {
  auto r = std::ref(x);
  return r.get() * r.get();
}

// Constructing the wrapper directly instead of through std::ref.
double f_direct_ctor(double x) {
  std::reference_wrapper<double> r(x);
  return r.get() * r.get();
}

// Writing through the wrapper, so the adjoint flows back out of it.
double f_write_through(double x) {
  double y = 0;
  std::reference_wrapper<double> r(y);
  r.get() = x * x;
  return y;
}

// Implicit conversion out of a const wrapper.
double f_cref_implicit(double x) { return read_cref_implicit(std::cref(x)); }

// Wrapping a user-defined type rather than a scalar.
double f_struct_ref(double x) {
  Holder h{x};
  std::reference_wrapper<Holder> r(h);
  return r.get().a * r.get().a;
}

// CHECK: void read_ref_pullback
// CHECK: get_reverse_forw
// CHECK: void f_ref_grad(double x, double *_d_x) {
// CHECK: clad::custom_derivatives::std::ref_reverse_forw

// CHECK: void read_cref_pullback
// CHECK: get_reverse_forw
// CHECK: void f_cref_grad(double x, double *_d_x) {
// CHECK: clad::custom_derivatives::std::cref_reverse_forw

// CHECK: void f_implicit_grad(double x, double *_d_x) {
// CHECK: clad::custom_derivatives::std::ref_reverse_forw

// CHECK: void f_mul_refs_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: clad::custom_derivatives::std::ref_reverse_forw

// The adjoint wrapper is copy-initialized from the propagator; reference_wrapper
// has no default constructor to fall back on.
// CHECK: void f_local_ref_grad(double x, double *_d_x) {
// CHECK: {{(std::)?}}reference_wrapper<double> _d_r = _t1.adjoint;

// CHECK: void f_direct_ctor_grad(double x, double *_d_x) {
// CHECK: {{(std::)?}}reference_wrapper<double> _d_r(*_d_x);

// CHECK: void f_write_through_grad(double x, double *_d_x) {
// CHECK: {{(std::)?}}reference_wrapper<double> _d_r(_d_y);

// A const wrapper still hands back a mutable adjoint for the referent.
// CHECK: void read_cref_implicit_pullback
// CHECK: clad::ValueAndAdjoint<const double &, double &> {{.*}} = {{.*}}conversion_operator_reverse_forw(&r, clad::Tag<{{(const double|type) &}}>(), _d_r);
// CHECK: void f_cref_implicit_grad(double x, double *_d_x) {
// CHECK: clad::custom_derivatives::std::cref_reverse_forw

// CHECK: void f_struct_ref_grad(double x, double *_d_x) {
// CHECK: {{(std::)?}}reference_wrapper<Holder> _d_r(_d_h);

int main() {
  double dx = 0;

  auto g_ref = clad::gradient(f_ref);
  g_ref.execute(3.0, &dx);
  printf("ref: %.4f\n", dx); // CHECK-EXEC: ref: 6.0000

  dx = 0;
  auto g_cref = clad::gradient(f_cref);
  g_cref.execute(4.0, &dx);
  printf("cref: %.4f\n", dx); // CHECK-EXEC: cref: 8.0000

  dx = 0;
  auto g_implicit = clad::gradient(f_implicit);
  g_implicit.execute(5.0, &dx);
  printf("implicit: %.4f\n", dx); // CHECK-EXEC: implicit: 10.0000

  dx = 0;
  auto g_twice = clad::gradient(f_twice_ref);
  g_twice.execute(2.0, &dx);
  printf("twice: %.4f\n", dx); // CHECK-EXEC: twice: 8.0000

  dx = 0;
  double dy = 0;
  auto g_sum = clad::gradient(f_sum_refs);
  g_sum.execute(1.0, 2.0, &dx, &dy);
  printf("sum: %.4f %.4f\n", dx, dy); // CHECK-EXEC: sum: 2.0000 4.0000

  dx = 0;
  dy = 0;
  auto g_mul = clad::gradient(f_mul_refs);
  g_mul.execute(3.0, 4.0, &dx, &dy);
  printf("mul: %.4f %.4f\n", dx, dy); // CHECK-EXEC: mul: 4.0000 3.0000

  dx = 0;
  auto g_local = clad::gradient(f_local_ref);
  g_local.execute(3.0, &dx);
  printf("local: %.4f\n", dx); // CHECK-EXEC: local: 6.0000

  dx = 0;
  auto g_ctor = clad::gradient(f_direct_ctor);
  g_ctor.execute(3.0, &dx);
  printf("ctor: %.4f\n", dx); // CHECK-EXEC: ctor: 6.0000

  dx = 0;
  auto g_write = clad::gradient(f_write_through);
  g_write.execute(3.0, &dx);
  printf("write: %.4f\n", dx); // CHECK-EXEC: write: 6.0000

  dx = 0;
  auto g_cref_impl = clad::gradient(f_cref_implicit);
  g_cref_impl.execute(4.0, &dx);
  printf("cref_implicit: %.4f\n", dx); // CHECK-EXEC: cref_implicit: 8.0000

  dx = 0;
  auto g_struct = clad::gradient(f_struct_ref);
  g_struct.execute(3.0, &dx);
  printf("struct: %.4f\n", dx); // CHECK-EXEC: struct: 6.0000

  return 0;
}
