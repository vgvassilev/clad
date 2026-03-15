// RUN: %cladclang %s -I%S/../../include -oStructReference.out | %filecheck %s
// RUN: ./StructReference.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

struct T {
  double x{};
  double y{};
};

double fn1(T& t) {
  return t.x * t.y;
}

double fn2(T&& t) {
  return t.x * t.y;
}

int main() {
  INIT_DIFFERENTIATE(fn1, "t.x")
  INIT_DIFFERENTIATE(fn2, "t.y")

  T t{2, 3};

  TEST_DIFFERENTIATE(fn1, t); // CHECK-EXEC: {3.00}
  TEST_DIFFERENTIATE(fn2, T{4, 5}); // CHECK-EXEC: {4.00}
  }
