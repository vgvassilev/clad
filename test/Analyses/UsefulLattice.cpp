// RUN: %cladclang -O0 %s -I%S/../../include -o %t.base > %t.base.codegen
// RUN: %filecheck --check-prefix=CHECK-BASE %s < %t.base.codegen
// RUN: %t.base | %filecheck_exec %s
// RUN: %cladclang -O0 -Xclang -plugin-arg-clad -Xclang -enable-ua %s -I%S/../../include -o %t.ua > %t.ua.codegen
// RUN: %filecheck --check-prefix=CHECK-UA %s < %t.ua.codegen
// RUN: %t.ua | %filecheck_exec %s
// RUN: %cladclang -O3 -DNDEBUG %s -I%S/../../include -o %t.opt > %t.opt.codegen
// RUN: %filecheck --check-prefix=CHECK-BASE %s < %t.opt.codegen
// RUN: %t.opt | %filecheck_exec %s
// RUN: %cladclang -O3 -DNDEBUG -Xclang -plugin-arg-clad -Xclang -enable-ua %s -I%S/../../include -o %t.ua-opt > %t.ua-opt.codegen
// RUN: %filecheck --check-prefix=CHECK-UA %s < %t.ua-opt.codegen
// RUN: %t.ua-opt | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

double carry(double x, int n) {
  double a = x, b = 0, c = 0;
  for (int i = 0; i < n; ++i) {
    c = b;
    b = a;
    a = x;
  }
  return c;
}

// CHECK-BASE-LABEL: double carry_darg0(double x, int n) {
// CHECK-BASE: int _d_i = 0;
// CHECK-BASE-LABEL: double arms_darg0

// CHECK-UA-LABEL: double carry_darg0(double x, int n) {
// CHECK-UA-NOT: _d_i
// CHECK-UA-LABEL: double arms_darg0

double arms(double x, double y, bool first) {
  double r = 0;
  if (first) {
    for (int i = 0; i < 3; ++i)
      r += x * x;
  } else {
    for (int i = 0; i < 3; ++i)
      r += y * y;
  }
  return r;
}

double nested(double x) {
  double r = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      r += x * x;
  return r;
}

double do_loop(double x, int n) {
  double r = 0;
  do {
    r += x;
  } while (--n > 0);
  return r;
}

double exits(double x) {
  double r = 0;
  for (int i = 0; i < 4; ++i) {
    if (i == 1)
      continue;
    r += x;
    if (i == 2)
      break;
  }
  return r;
}

double pointer_local(double x) {
  double a[2] = {x, x * x};
  double* p = a;
  return p[0] + p[1];
}

double reference_local(double x) {
  double a = x * x;
  double& r = a;
  return r;
}

struct Pair {
  double a;
  double b;
};

double record_local(double x) {
  Pair p{x, x * x};
  return p.a + p.b;
}

static void check(const char* name, double actual, double expected) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > 1e-12 * (1 + std::fabs(expected))) {
    std::fprintf(stderr, "%s: got %.17g, expected %.17g\n", name, actual,
                 expected);
    std::exit(1);
  }
}

int main() {
  auto dc = clad::differentiate(carry, "x");
  auto dax = clad::differentiate(arms, "x");
  auto day = clad::differentiate(arms, "y");
  auto dn = clad::differentiate(nested, "x");
  auto dd = clad::differentiate(do_loop, "x");
  auto de = clad::differentiate(exits, "x");
  auto dp = clad::differentiate(pointer_local, "x");
  auto dr = clad::differentiate(reference_local, "x");
  auto ds = clad::differentiate(record_local, "x");
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (double x : {-0.5, 0.0, 2.0}) {
      for (int n : {0, 1, 2, 5})
        check("carry", dc.execute(x, n), n < 2 ? 0.0 : 1.0);
      for (bool first : {false, true}) {
        check("arms/x", dax.execute(x, 0.5, first), first ? 6 * x : 0);
        check("arms/y", day.execute(x, 0.5, first), first ? 0 : 3);
      }
      check("nested", dn.execute(x), 12 * x);
      for (int n : {1, 3})
        check("do_loop", dd.execute(x, n), n);
      check("exits", de.execute(x), 2);
      check("pointer", dp.execute(x), 1 + 2 * x);
      check("reference", dr.execute(x), 2 * x);
      check("record", ds.execute(x), 1 + 2 * x);
    }
  }
  std::puts("Useful lattice derivatives OK");
  // CHECK-EXEC: Useful lattice derivatives OK
}
