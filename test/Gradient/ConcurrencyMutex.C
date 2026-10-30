// RUN: %cladclang %s -I%S/../../include -lpthread -std=c++17 -oConcurrencyMutex.out 2>&1 | %filecheck %s
// RUN: ./ConcurrencyMutex.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <cstdio>
#include <mutex>

double f_mutex_sq(double x) {
  std::mutex m;
  m.lock();
  double y = x * x;
  m.unlock();
  return y;
}

// Tape try_lock's bool; the reverse sweep must not lock again.
double f_mutex_try_lock(double x) {
  std::mutex m;
  double y = x;
  bool ok = m.try_lock();
  if (ok) {
    y = x * x;
    m.unlock();
  }
  return y;
}

// CHECK: void f_mutex_sq_grad(double x, double *_d_x) {
// CHECK-NOT: constructor_reverse_forw
// CHECK: class_functions::lock_reverse_forw
// CHECK: class_functions::unlock_reverse_forw
// CHECK: void f_mutex_try_lock_grad(double x, double *_d_x) {
// CHECK: class_functions::try_lock_reverse_forw

int main() {
  double dx = 0;

  auto g_sq = clad::gradient(f_mutex_sq);
  g_sq.execute(3.0, &dx);
  printf("mutex_sq: %.4f\n", dx); // CHECK-EXEC: mutex_sq: 6.0000

  dx = 0;
  auto g_try = clad::gradient(f_mutex_try_lock);
  g_try.execute(3.0, &dx);
  printf("try_lock: %.4f\n", dx); // CHECK-EXEC: try_lock: 6.0000

  return 0;
}
