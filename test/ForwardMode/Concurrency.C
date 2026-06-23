// RUN: %cladclang %s -I%S/../../include -oConcurrency.out 2>&1 | %filecheck %s
// RUN: ./Concurrency.out | %filecheck_exec %s
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <cstdio>
#include <functional>

// --- helpers ---

double read_ref(std::reference_wrapper<double> r) { return r.get() * r.get(); }

double read_cref(std::reference_wrapper<const double> r) { return r.get() * r.get(); }

double read_implicit(std::reference_wrapper<double> r) {
  double& v = r;
  return v * v;
}

double mul_refs(std::reference_wrapper<double> a,
                std::reference_wrapper<double> b) {
  return a.get() * b.get();
}

// --- forward-mode functions ---

double f_ref(double x) { return read_ref(std::ref(x)); }

double f_cref(double x) { return read_cref(std::cref(x)); }

double f_implicit(double x) { return read_implicit(std::ref(x)); }

double f_twice_ref(double x) {
  return read_ref(std::ref(x)) + read_ref(std::ref(x));
}

double f_sum_refs(double x, double y) {
  return read_ref(std::ref(x)) + read_ref(std::ref(y));
}

double f_mul_refs(double x, double y) { return mul_refs(std::ref(x), std::ref(y)); }

// CHECK: double f_ref_darg0(double x) {
// CHECK: double f_mul_refs_darg0(double x, double y) {

int main() {
  auto d_ref = clad::differentiate(f_ref, "x");
  printf("ref: %.4f\n", d_ref.execute(3.0)); // CHECK-EXEC: ref: 6.0000

  auto d_cref = clad::differentiate(f_cref, "x");
  printf("cref: %.4f\n", d_cref.execute(4.0)); // CHECK-EXEC: cref: 8.0000

  auto d_implicit = clad::differentiate(f_implicit, "x");
  printf("implicit: %.4f\n", d_implicit.execute(5.0)); // CHECK-EXEC: implicit: 10.0000

  auto d_twice = clad::differentiate(f_twice_ref, "x");
  printf("twice: %.4f\n", d_twice.execute(2.0)); // CHECK-EXEC: twice: 8.0000

  auto d_sum_x = clad::differentiate(f_sum_refs, "x");
  printf("sum_x: %.4f\n", d_sum_x.execute(1.0, 2.0)); // CHECK-EXEC: sum_x: 2.0000

  auto d_sum_y = clad::differentiate(f_sum_refs, "y");
  printf("sum_y: %.4f\n", d_sum_y.execute(1.0, 2.0)); // CHECK-EXEC: sum_y: 4.0000

  auto d_mul_x = clad::differentiate(f_mul_refs, "x");
  printf("mul_x: %.4f\n", d_mul_x.execute(3.0, 4.0)); // CHECK-EXEC: mul_x: 4.0000

  auto d_mul_y = clad::differentiate(f_mul_refs, "y");
  printf("mul_y: %.4f\n", d_mul_y.execute(3.0, 4.0)); // CHECK-EXEC: mul_y: 3.0000

  return 0;
}
