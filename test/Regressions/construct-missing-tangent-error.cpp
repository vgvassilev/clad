// RUN: %cladclang -fsyntax-only -Xclang -verify -std=c++17 -I%S/../../include %s

// Constructions forward mode cannot derive when an argument has no tangent.

#include "clad/Differentiator/Differentiator.h"

// A pointer to differentiable data: neither null nor the primal pointer is
// its tangent.
struct WP {
  double* q;
  WP(double* q) : q(q) {}
};
double* gq;
double pointer(double x) {
  double a = x;
  gq = &a;
  WP w(gq); // expected-error {{cannot derive this construction: the pointer argument has no derivative}}
  return *w.q * x;
}

// Built from data without a derivative, but with no zero value to stand for
// that of the object.
struct Tag {
  char c;
  Tag(const char* s) : c(s[0]) {}
};
double first(const Tag& t) { return t.c; }
struct Drift { static constexpr const char* type = "Drift"; };
double tag(double x) {
  return first(Tag(Drift::type)) * x; // expected-error {{cannot derive this construction: its arguments have no derivative and 'Tag' has no zero value}}
}

// An object argument without a tangent and without a zero value.
struct V {
  double a;
  V(double a) : a(a) {}
};
struct Holder { V v; };
Holder holder{V(5)};
Holder* hp = &holder;
struct W {
  double val;
  W(const V& r) : val(r.a) {}
};
double object(double x) {
  W w(hp->v); // expected-warning {{'v' is accessed through an expression without a derivative; its derivative is treated as 0}} \
              // expected-error {{cannot derive this construction: the argument has no derivative and 'V' has no zero value}}
  return w.val * x;
}

// Through a pointer to non-const the constructor's increment would run twice.
int counter = 0;
int* cp = &counter;
struct Counted {
  double v;
  Counted(double x, int* c) : v(x) { ++*c; }
};
double counted(double x) {
  Counted c(x, cp); // expected-error {{the derivative of this construction is built on the same pointer; what its constructor and destructor do through it happens twice; pass a pointer to const if it only reads through it}}
  return c.v * x;
}

// The differentiable data may sit in a base class.
struct DataBase { double d; };
struct Data : DataBase { int n; };
Data data{};
const Data* dp = &data;
struct FromData {
  double v;
  FromData(double x, const Data* p) : v(x) {}
};
double base(double x) {
  FromData f(x, dp); // expected-error {{cannot derive this construction: the pointer argument has no derivative}}
  return f.v * x;
}

void test() {
  clad::differentiate(pointer, "x");
  clad::differentiate(counted, "x");
  clad::differentiate(base, "x");
  clad::differentiate(tag, "x");
  clad::differentiate(object, "x");
}
