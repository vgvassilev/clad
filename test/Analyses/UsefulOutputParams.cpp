// RUN: %cladclang %s -I%S/../../include -oUsefulOutputParams.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./UsefulOutputParams.out | %filecheck_exec %s

// A function that hands back its result through a reference has an output,
// as one that returns it does: what it computes for it is useful, and keeps
// its tangent. A call that writes a useful variable through an argument
// reads all of its arguments to do so: they are useful too, as are those of a
// method that changes a useful object. An assignment to what such a call
// returns writes a useful variable too.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

void ratio(double a, double b, double& r) {
  double t = a / b;
  r = t * 2;
}

double f(double a, double b) {
  double r;
  ratio(a, b, r);
  return r;
}

// CHECK: void ratio_pushforward(double a, double b, double &r, double _d_a, double _d_b, double &_d_r) {
// CHECK-NEXT: double _d_t = (_d_a * b - a * _d_b) / (b * b);
// CHECK-NEXT: double t = a / b;
// CHECK-NEXT: _d_r = _d_t * 2;

struct Element {
  double k;
};

void scale(const Element& e, double& c) { c = c * e.k; }

double run(double k, double x) {
  Element e{k};
  double c = x;
  scale(e, c);
  return c;
}

double g(double k, double x) { return run(k, x); }

// CHECK: clad::ValueAndPushforward<double, double> run_pushforward(double k, double x, double _d_k, double _d_x) {
// CHECK-NEXT: Element _d_e{_d_k};
// CHECK: scale_pushforward(e, c, _d_e, _d_c);

struct Pair {
  double v[2];
  double& operator()(int i) { return v[i]; }
};

Pair twice(double s) {
  Pair p{{0, 0}};
  double t = s * 2;
  p(1) = t;
  return p;
}

double h(double s) { return twice(s).v[1]; }

// CHECK: clad::ValueAndPushforward<Pair, Pair> twice_pushforward(double s, double _d_s) {
// CHECK: double _d_t = _d_s * 2;
// CHECK-NEXT: double t = s * 2;
// CHECK-NEXT: {{.*}} _t0 = p.operator_call_pushforward(1, &_d_p, 0);
// CHECK-NEXT: _t0.pushforward = _d_t;

struct Acc {
  double v;
  void add(double t) { v += t; }
};

double sum(double s) {
  Acc a{0};
  double t = s * 2;
  a.add(t);
  return a.v;
}

// CHECK: double sum_darg0(double s) {
// CHECK: double _d_t = _d_s * 2;
// CHECK-NEXT: double t = s * 2;
// CHECK-NEXT: a.add_pushforward(t, &_d_a, _d_t);

int main() {
  auto d = clad::differentiate<clad::opts::enable_ua>(f, "a");
  printf("%.2f\n", d.execute(3, 2)); // CHECK-EXEC: 1.00
  auto dk = clad::differentiate<clad::opts::enable_ua>(g, "k");
  printf("%.2f\n", dk.execute(2, 3)); // CHECK-EXEC: 3.00
  auto ds = clad::differentiate<clad::opts::enable_ua>(h, "s");
  printf("%.2f\n", ds.execute(3)); // CHECK-EXEC: 2.00
  auto dsum = clad::differentiate<clad::opts::enable_ua>(sum, "s");
  printf("%.2f\n", dsum.execute(3)); // CHECK-EXEC: 2.00
}
