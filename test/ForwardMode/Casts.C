// RUN: %cladclang %s -I%S/../../include -oCasts.out 2>&1 | %filecheck %s
// RUN: ./Casts.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

#include <utility>

struct S {
  float get() { return 3.; }
};
long double fn1(double i, double j) {
  long double res =
      static_cast<long double>(7 * i) + static_cast<long double>(i * j);
  const S * s_const = new S();
  S * s = const_cast<S*>(s_const);
  long double x = reinterpret_cast<S*>(s)->get() + dynamic_cast<S*>(s)->get();
  delete s;
  return res + x;
}

// std::move and std::forward only cast their argument, and so the tangent of
// what they return is that of the argument.
struct D {
  double v;
};

double moved(double x) {
  D d{x * x};
  const D& r = std::move(d);
  return r.v * x;
}

template <class T> double square(T&& t) {
  double&& u = std::forward<T>(t);
  return u * u;
}

double forwarded(double x) { return square(x * x); }

int main() {
  INIT_DIFFERENTIATE(fn1, "i");

  TEST_DIFFERENTIATE(fn1, 3, 5);  // CHECK-EXEC: {12.00}

  INIT_DIFFERENTIATE(moved, "x");
  INIT_DIFFERENTIATE(forwarded, "x");
  TEST_DIFFERENTIATE(moved, 3);     // CHECK-EXEC: {27.00}
  TEST_DIFFERENTIATE(forwarded, 3); // CHECK-EXEC: {108.00}
}
