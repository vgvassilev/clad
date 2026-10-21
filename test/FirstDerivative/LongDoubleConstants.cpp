// RUN: %cladclang %s -I%S/../../include -oLongDoubleConstants.out
// RUN: ./LongDoubleConstants.out | %filecheck_exec %s

// The derivatives of exp2, log2, log10, erf and erfc carry ln 2, ln 10 or
// 2/sqrt(pi). Check that a long double derivative carries them to long double
// precision, not to double's, which is hundreds of ulps off in x87 long double
// and far more in quad. Also check that the float derivative of erfc compiles:
// it used to narrow a double into the float pushforward's braced return.

#include "clad/Differentiator/Differentiator.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

long double f_exp2(long double x) { return std::exp2(x); }
long double f_log2(long double x) { return std::log2(x); }
long double f_log10(long double x) { return std::log10(x); }
long double f_erf(long double x) { return std::erf(x); }
long double f_erfc(long double x) { return std::erfc(x); }
float f_erfcf(float x) { return std::erfc(x); }

const long double Ln2 = 0.693147180559945309417232121458176568075500L;
const long double Ln10 = 2.302585092994045684017991454684364207601101L;
const long double TwoOverSqrtPi = 1.128379167095512573896158903121545171688L;

// The epsilon of long double as the machine computes it. Valgrind keeps long
// double in 64 bits, and there no derivative can do better than double.
long double epsilon_in_use() {
  volatile long double e = 1;
  volatile long double sum = 2;
  while (true) {
    sum = 1 + e / 2;
    if (sum == 1)
      return e;
    e = e / 2;
  }
}

void check(const char* name, long double got, long double want) {
  long double tol = 8 * std::max(std::numeric_limits<long double>::epsilon(),
                                 epsilon_in_use());
  bool ok = std::fabs(got - want) <= tol * std::fabs(want);
  std::printf("%s: %s\n", name, ok ? "ok" : "off");
}

int main() {
  long double x = 1.5L;
  auto d_exp2 = clad::differentiate(f_exp2, "x");
  auto d_log2 = clad::differentiate(f_log2, "x");
  auto d_log10 = clad::differentiate(f_log10, "x");
  auto d_erf = clad::differentiate(f_erf, "x");
  auto d_erfc = clad::differentiate(f_erfc, "x");

  check("exp2", d_exp2.execute(x), std::exp2(x) * Ln2);
  // CHECK-EXEC: exp2: ok
  check("log2", d_log2.execute(x), 1 / (x * Ln2));
  // CHECK-EXEC: log2: ok
  check("log10", d_log10.execute(x), 1 / (x * Ln10));
  // CHECK-EXEC: log10: ok
  check("erf", d_erf.execute(x), TwoOverSqrtPi * std::exp(-x * x));
  // CHECK-EXEC: erf: ok
  check("erfc", d_erfc.execute(x), -TwoOverSqrtPi * std::exp(-x * x));
  // CHECK-EXEC: erfc: ok

  auto d_erfcf = clad::differentiate(f_erfcf, "x");
  float y = 1.5F;
  float got = d_erfcf.execute(y);
  float want = -1.12837917F * std::exp(-y * y);
  bool ok = std::fabs(got - want) <= 8 * std::numeric_limits<float>::epsilon() *
                                         std::fabs(want);
  std::printf("erfc float: %s\n", ok ? "ok" : "off");
  // CHECK-EXEC: erfc float: ok
}
