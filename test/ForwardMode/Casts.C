// RUN: %cladclang %s -I%S/../../include -oCasts.out 2>&1 | %filecheck %s
// RUN: ./Casts.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"
struct S {
  float get() { return 3.; }
};
long double fn1(double i, double j) {
  long double res =
      static_cast<long double>(7 * i) + static_cast<long double>(i * j);
  const S * s_const = new S();
  S * s = const_cast<S*>(s_const);
  long double x = reinterpret_cast<S*>(s)->get() + dynamic_cast<S*>(s)->get();
  return res + x;
}

int main() {
  INIT_DIFFERENTIATE(fn1, "i");

  TEST_DIFFERENTIATE(fn1, 3, 5);  // CHECK-EXEC: {12.00}
}
