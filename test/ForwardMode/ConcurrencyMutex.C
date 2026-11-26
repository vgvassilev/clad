// RUN: %cladclang %s -I%S/../../include -lpthread -std=c++17 -oConcurrencyMutex.out 2>&1 | %filecheck %s
// RUN: ./ConcurrencyMutex.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <cstdio>
#include <mutex>

double f_mutex_lock(double x) {
  std::mutex m;
  double y = x;
  m.lock();
  y *= 2.0;
  m.unlock();
  return y;
}

double f_mutex_try_lock(double x) {
  std::mutex m;
  double y = x;
  bool ok = m.try_lock();
  if (ok) {
    y *= 2.0;
    m.unlock();
  }
  return y;
}

// CHECK: double f_mutex_lock_darg0(double x) {
// CHECK: class_functions::lock_pushforward
// CHECK: class_functions::unlock_pushforward
// CHECK: double f_mutex_try_lock_darg0(double x) {
// CHECK: class_functions::try_lock_pushforward

int main() {
  auto d_mutex = clad::differentiate(f_mutex_lock, "x");
  printf("mutex: %.4f\n", d_mutex.execute(3.0)); // CHECK-EXEC: mutex: 2.0000

  auto d_trylock = clad::differentiate(f_mutex_try_lock, "x");
  printf("try_lock: %.4f\n", d_trylock.execute(3.0)); // CHECK-EXEC: try_lock: 2.0000

}
