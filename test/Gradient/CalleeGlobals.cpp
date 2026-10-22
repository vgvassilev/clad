// RUN: %cladclang %s -I%S/../../include -oCalleeGlobals.out \
// RUN:   -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./CalleeGlobals.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=all %s \
// RUN:   -I%S/../../include -oCalleeGlobals.out -Xclang -verify
// RUN: ./CalleeGlobals.out | %filecheck_exec %s

// A callee that reads a global the differentiated code assigns, or assigns
// one it reads, moves a derivative through the global's adjoint, so it needs
// a pullback whatever its arguments.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>
#include <string>

double g = 0; // expected-warning {{gradient uses a global variable 'g'; rerunning the gradient requires 'g' to be reset}}

double read_g() { return g * g; }
// f = x^2, f' = 2x.
double direct(double x) {
  g = x;
  return read_g();
}

// The read is two calls further down. f = 2x^2, f' = 4x.
double leaf() { return g * g; }
double mid() { return 2 * leaf(); }
double chain(double x) {
  g = x;
  return mid();
}

// A real function of one real argument is otherwise differentiated by its
// pushforward. f = k x^2 with k = 2 or k = x.
double scaled(double k) { return k * g * g; }
double one_arg_literal(double x) {
  g = x;
  return scaled(2.0); // f' = 4x
}
double one_arg_varied(double x) {
  g = x;
  return scaled(x); // f' = 3x^2
}

// A constructor reads it in its member initializers. f = x^2.
struct FromG {
  double v;
  FromG() : v(g * g) {}
};
double ctor(double x) {
  g = x;
  FromG r;
  return r.v; // f' = 2x
}
// A struct of doubles has an adjoint too. f = x^2, f' = 2x.
struct Pair {
  double a, b;
} gp; // expected-warning {{gradient uses a global variable 'gp'; rerunning the gradient requires 'gp' to be reset}}
double read_gp() { return gp.a * gp.a + gp.b; }
double member(double x) {
  gp.a = x;
  return read_gp();
}

// A class with a constructor is not a plain aggregate: no adjoint, and the
// call stays out. f = x, f' = 1.
struct Counter {
  int n;
  Counter() : n(0) {}
} counter; // expected-warning {{gradient uses a global variable 'counter'; rerunning the gradient requires 'counter' to be reset}}
double count() {
  ++counter.n;
  return 1.0;
}
double counted(double x) {
  counter.n = 0;
  return x * count();
}

// Only stores into g: no read, nothing to carry. f = x, f' = 1.
double store_g() {
  g = 0.5;
  return 1.0;
}
double store_only(double x) {
  g = x;
  return x * store_g();
}

// A pointer has no adjoint to carry, even one the differentiated function
// assigns: the call stays out. f = 2x, f' = 2.
double coefficient = 2; // expected-warning {{gradient uses a global variable 'coefficient'; rerunning the gradient requires 'coefficient' to be reset}}
double* tp = nullptr; // expected-warning {{gradient uses a global variable 'tp'; rerunning the gradient requires 'tp' to be reset}}
double lookup() { return *tp; }
double through_pointer(double x) {
  tp = &coefficient;
  return x * lookup();
}

// g is read but not assigned by the differentiated function, so the callee
// stays out. f' = 1.
double format_g() {
  std::string s = std::to_string(g);
  return s.empty() ? 0.0 : 1.0;
}
double unassigned(double x) { return x * format_g(); }

// The callee assigns g, which the caller reads afterwards: its pullback is
// what moves the derivative. f = x^2, f' = 2x.
void set_g(double v) { g = v; }
double setter(double x) {
  set_g(x);
  return g * g;
}
// One callee assigns g and another reads it; the caller does neither.
// f = x^2, f' = 2x.
double setter_then_reader(double x) {
  set_g(x);
  return read_g();
}
// A constant store kills what the caller had put in g. f = 5x, f' = 5.
void reset_g() { g = 5.0; }
double killer(double x) {
  g = x;
  reset_g();
  return g * x;
}

// A read through a global pointer that may alias g is not followed, and is
// reported.
double* pg = &g;
double read_through_pointer() { return *pg * *pg; }
double aliased(double x) {
  g = x;
  return read_through_pointer(); // expected-warning {{'read_through_pointer' reads the global 'pg' through a pointer or reference; clad does not differentiate through it}}
}

// `sizeof(g)` does not read g. f = 8x, f' = 8.
double size_of_g() { return sizeof(g); }
double unevaluated(double x) {
  g = x;
  return x * size_of_g();
}

// CHECK-NOT: lookup_pullback
// CHECK-NOT: size_of_g_pullback
// CHECK-NOT: format_g_pullback
// CHECK-NOT: store_g_pullback
// CHECK-NOT: count_pullback

#define TEST(F, want)                                                          \
  {                                                                            \
    double d = 0;                                                              \
    clad::gradient(F).execute(x, &d);                                          \
    printf("%s %.4f %.4f\n", #F, d, want);                                     \
  }

int main() {
  double x = 1.5;
  TEST(direct, 2 * x);           // CHECK-EXEC: direct 3.0000 3.0000
  TEST(chain, 4 * x);            // CHECK-EXEC: chain 6.0000 6.0000
  TEST(one_arg_literal, 4 * x);  // CHECK-EXEC: one_arg_literal 6.0000 6.0000
  TEST(one_arg_varied, 3 * x * x); // CHECK-EXEC: one_arg_varied 6.7500 6.7500
  TEST(ctor, 2 * x);             // CHECK-EXEC: ctor 3.0000 3.0000
  TEST(member, 2 * x);           // CHECK-EXEC: member 3.0000 3.0000
  TEST(counted, 1.0);           // CHECK-EXEC: counted 1.0000 1.0000
  TEST(store_only, 1.0);         // CHECK-EXEC: store_only 1.0000 1.0000
  TEST(through_pointer, 2.0);    // CHECK-EXEC: through_pointer 2.0000 2.0000
  TEST(unassigned, 1.0);         // CHECK-EXEC: unassigned 1.0000 1.0000
  TEST(setter, 2 * x);           // CHECK-EXEC: setter 3.0000 3.0000
  TEST(setter_then_reader, 2 * x); // CHECK-EXEC: setter_then_reader 3.0000 3.0000
  TEST(killer, 5.0);             // CHECK-EXEC: killer 5.0000 5.0000
  TEST(unevaluated, 8.0);        // CHECK-EXEC: unevaluated 8.0000 8.0000
  clad::gradient(aliased);
}
