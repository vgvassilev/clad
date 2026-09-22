// RUN: %cladclang %s -I%S/../../include -lpthread -std=c++17 -oConcurrencyThread.out 2>&1 | %filecheck %s
// RUN: ./ConcurrencyThread.out | %filecheck_exec %s
// XFAIL: valgrind

// Reverse-mode std::thread support.
// Covered: noop+join, pointer / std::ref shared workers, named functors
// (incl. multi-arg), multi-arg free functions, multiple threads, joinable.
// Not covered: lambda callables (diagnosed), detach (diagnosed; join required),
// parallel reverse sweeps.

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <cstdio>
#include <functional>
#include <thread>

void thread_noop() {}

double f_thread_noop(double x) {
  std::thread t(thread_noop);
  t.join();
  return x * x;
}

void set_y(double x, double* y) { *y = x * x; }

double f_thread_ptr(double x) {
  double y = 0.0;
  std::thread t(set_y, x, &y);
  t.join();
  return y;
}

void set_y_ref(double x, double& y) { y = x * x; }

double f_thread_ref(double x) {
  double y = 0.0;
  std::thread t(set_y_ref, x, std::ref(y));
  t.join();
  return y;
}

struct SquareTo {
  void operator()(double v, double* out) const { *out = v * v; }
};

double f_thread_functor(double x) {
  double y = 0.0;
  std::thread t(SquareTo{}, x, &y);
  t.join();
  return y;
}

struct ScaleByRef {
  void operator()(double& r) const { r *= 2.0; }
};

double f_thread_functor_ref(double x) {
  std::thread t(ScaleByRef{}, std::ref(x));
  t.join();
  return x;
}

void add2(double a, double b, double* out) { *out = a + b; }

double f_thread_two_args(double x, double y) {
  double z = 0.0;
  std::thread t(add2, x, y, &z);
  t.join();
  return z;
}

void add3_to_first(double& a, double b, double c) { a += b + c; }

double f_thread_three_args(double x, double y, double z) {
  std::thread t(add3_to_first, std::ref(x), y, z);
  t.join();
  return x;
}

void inc_ref(double& r) { r += 1.0; }

double f_thread_two(double x) {
  std::thread t1(inc_ref, std::ref(x));
  t1.join();
  std::thread t2(inc_ref, std::ref(x));
  t2.join();
  return x;
}

double f_thread_parallel(double x, double y) {
  std::thread t1(inc_ref, std::ref(x));
  std::thread t2(inc_ref, std::ref(y));
  t1.join();
  t2.join();
  return x + y;
}

double f_thread_joinable(double x) {
  std::thread t(inc_ref, std::ref(x));
  bool j = t.joinable();
  if (j)
    t.join();
  return x;
}

// CHECK: void f_thread_noop_grad(double x, double *_d_x) {
// CHECK: clad::custom_derivatives::class_functions::constructor_reverse_forw
// CHECK: clad::custom_derivatives::class_functions::join_reverse_forw

// CHECK: void f_thread_ptr_grad(double x, double *_d_x) {
// CHECK: set_y_pullback

// CHECK: void f_thread_ref_grad(double x, double *_d_x) {
// CHECK: clad::custom_derivatives::std::ref_reverse_forw
// CHECK: set_y_ref_pullback

// CHECK: void f_thread_functor_grad(double x, double *_d_x) {
// CHECK: operator_call_pullback

// CHECK: void f_thread_functor_ref_grad(double x, double *_d_x) {
// CHECK: operator_call_pullback

// CHECK: void f_thread_two_args_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: add2_pullback

// CHECK: void f_thread_three_args_grad(double x, double y, double z, double *_d_x, double *_d_y, double *_d_z) {
// CHECK: add3_to_first_pullback

// CHECK: void f_thread_two_grad(double x, double *_d_x) {
// CHECK: inc_ref_pullback

// CHECK: void f_thread_parallel_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: inc_ref_pullback

// CHECK: void f_thread_joinable_grad(double x, double *_d_x) {
// CHECK: t.joinable()
// CHECK: clad::custom_derivatives::class_functions::join_reverse_forw

int main() {
  double dx = 0;

  auto g_noop = clad::gradient(f_thread_noop);
  g_noop.execute(3.0, &dx);
  printf("thread_noop: %.4f\n", dx); // CHECK-EXEC: thread_noop: 6.0000

  dx = 0;
  auto g_ptr = clad::gradient(f_thread_ptr);
  g_ptr.execute(3.0, &dx);
  printf("thread_ptr: %.4f\n", dx); // CHECK-EXEC: thread_ptr: 6.0000

  dx = 0;
  auto g_ref = clad::gradient(f_thread_ref);
  g_ref.execute(3.0, &dx);
  printf("thread_ref: %.4f\n", dx); // CHECK-EXEC: thread_ref: 6.0000

  dx = 0;
  auto g_fun = clad::gradient(f_thread_functor);
  g_fun.execute(3.0, &dx);
  printf("thread_functor: %.4f\n", dx); // CHECK-EXEC: thread_functor: 6.0000

  dx = 0;
  auto g_fun_ref = clad::gradient(f_thread_functor_ref);
  g_fun_ref.execute(3.0, &dx);
  printf("thread_functor_ref: %.4f\n", dx); // CHECK-EXEC: thread_functor_ref: 2.0000

  dx = 0;
  double dy = 0;
  auto g_two_args = clad::gradient(f_thread_two_args);
  g_two_args.execute(3.0, 4.0, &dx, &dy);
  printf("thread_2arg: %.4f %.4f\n", dx, dy); // CHECK-EXEC: thread_2arg: 1.0000 1.0000

  dx = 0;
  dy = 0;
  double dz = 0;
  auto g_three = clad::gradient(f_thread_three_args);
  g_three.execute(1.0, 2.0, 3.0, &dx, &dy, &dz);
  printf("thread_3arg: %.4f %.4f %.4f\n", dx, dy, dz); // CHECK-EXEC: thread_3arg: 1.0000 1.0000 1.0000

  dx = 0;
  auto g_two = clad::gradient(f_thread_two);
  g_two.execute(3.0, &dx);
  printf("thread_two: %.4f\n", dx); // CHECK-EXEC: thread_two: 1.0000

  dx = 0;
  dy = 0;
  auto g_par = clad::gradient(f_thread_parallel);
  g_par.execute(1.0, 2.0, &dx, &dy);
  printf("thread_par: %.4f %.4f\n", dx, dy); // CHECK-EXEC: thread_par: 1.0000 1.0000

  dx = 0;
  auto g_join = clad::gradient(f_thread_joinable);
  g_join.execute(2.0, &dx);
  printf("thread_joinable: %.4f\n", dx); // CHECK-EXEC: thread_joinable: 1.0000

  return 0;
}
