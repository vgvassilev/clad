// RUN: %cladclang %s -I%S/../../include -lpthread -std=c++17 -oConcurrency.out 2>&1 | %filecheck %s
// RUN: ./Concurrency.out | %filecheck_exec %s
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <cstdio>
#include <functional>
#include <thread>

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

void inc_ref(std::reference_wrapper<double> r) { r.get() += 1.0; }

void scale_ref(std::reference_wrapper<double> r) { r.get() *= 2.0; }

void add2_refs(std::reference_wrapper<double> a,
               std::reference_wrapper<double> b) {
  a.get() += b.get();
}

double f_thread_ref(double x) {
  std::reference_wrapper<double> rx = std::ref(x);
  std::thread t(inc_ref, rx);
  t.join();
  return x;
}

double f_thread_square(double x) {
  std::reference_wrapper<double> rx = std::ref(x);
  std::thread t(inc_ref, rx);
  t.join();
  return x * x;
}

double f_thread_scale(double x) {
  std::reference_wrapper<double> rx = std::ref(x);
  std::thread t(scale_ref, rx);
  t.join();
  return x;
}

double f_thread_joinable(double x) {
  std::reference_wrapper<double> rx = std::ref(x);
  std::thread t(inc_ref, rx);
  bool j = t.joinable();
  if (j)
    t.join();
  return x;
}

double f_thread_two(double x) {
  std::reference_wrapper<double> rx = std::ref(x);
  std::thread t1(inc_ref, rx);
  t1.join();
  std::thread t2(inc_ref, rx);
  t2.join();
  return x;
}

double f_thread_parallel(double x, double y) {
  std::reference_wrapper<double> rx = std::ref(x);
  std::reference_wrapper<double> ry = std::ref(y);
  std::thread t1(inc_ref, rx);
  std::thread t2(inc_ref, ry);
  t1.join();
  t2.join();
  return x + y;
}

double f_thread_two_args(double x, double y) {
  std::reference_wrapper<double> rx = std::ref(x);
  std::reference_wrapper<double> ry = std::ref(y);
  std::thread t(add2_refs, rx, ry);
  t.join();
  return x + y;
}

// CHECK: double f_ref_darg0(double x) {
// CHECK: double f_mul_refs_darg0(double x, double y) {
// CHECK: double f_thread_ref_darg0(double x) {
// CHECK: double f_thread_square_darg0(double x) {
// CHECK: double f_thread_two_args_darg0(double x, double y) {

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

  auto d_thread = clad::differentiate(f_thread_ref, "x");
  printf("thread: %.4f\n", d_thread.execute(2.0)); // CHECK-EXEC: thread: 1.0000

  auto d_sq = clad::differentiate(f_thread_square, "x");
  printf("thread_sq: %.4f\n", d_sq.execute(2.0)); // CHECK-EXEC: thread_sq: 6.0000

  auto d_scale = clad::differentiate(f_thread_scale, "x");
  printf("thread_scale: %.4f\n", d_scale.execute(3.0)); // CHECK-EXEC: thread_scale: 2.0000

  auto d_join = clad::differentiate(f_thread_joinable, "x");
  printf("thread_join: %.4f\n", d_join.execute(2.0)); // CHECK-EXEC: thread_join: 1.0000

  auto d_two = clad::differentiate(f_thread_two, "x");
  printf("thread_two: %.4f\n", d_two.execute(2.0)); // CHECK-EXEC: thread_two: 1.0000

  auto d_par_x = clad::differentiate(f_thread_parallel, "x");
  printf("thread_par_x: %.4f\n", d_par_x.execute(1.0, 2.0)); // CHECK-EXEC: thread_par_x: 1.0000

  auto d_par_y = clad::differentiate(f_thread_parallel, "y");
  printf("thread_par_y: %.4f\n", d_par_y.execute(1.0, 2.0)); // CHECK-EXEC: thread_par_y: 1.0000

  auto d_two_args_x = clad::differentiate(f_thread_two_args, "x");
  printf("thread_2arg_x: %.4f\n", d_two_args_x.execute(1.0, 2.0)); // CHECK-EXEC: thread_2arg_x: 1.0000

  auto d_two_args_y = clad::differentiate(f_thread_two_args, "y");
  printf("thread_2arg_y: %.4f\n", d_two_args_y.execute(1.0, 2.0)); // CHECK-EXEC: thread_2arg_y: 2.0000

  return 0;
}
