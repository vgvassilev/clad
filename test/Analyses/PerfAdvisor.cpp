// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify \
// RUN:   -Xclang -plugin-arg-clad -Xclang -Rclad-perf
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify=none

// none-no-diagnostics

// -Rclad-perf says where a large share of a gradient's work goes to what a
// hand-written derivative would not do, and what to write instead.

#include "clad/Differentiator/Differentiator.h"

struct M {
  double a[6][6];
  double& operator()(int i, int j) { return a[i - 1][j - 1]; }
  double operator()(int i, int j) const { return a[i - 1][j - 1]; }
  static M Identity() {
    M r{};
    for (int i = 0; i < 6; ++i)
      r.a[i][i] = 1;
    return r;
  }
  M t() const {
    M r{};
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        r.a[i][j] = a[j][i];
    return r;
  }
};

// Its pullback runs these loops again before running them backwards.
M mul(const M& x, const M& y) { // expected-remark-re {{the pullback of 'mul' runs its forward work again, about {{[0-9]+}}% of the work of the gradient of 'track'}} expected-note {{it runs the loops and arithmetic of its own body again}} expected-note {{a hand-written 'mul_reverse_forw' that hands its pullback what it needs through clad::pullback_state runs it once}}
  M r{};
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < 6; ++k)
        r.a[i][j] += x.a[i][k] * y.a[k][j];
  return r;
}

M keep(const M& m) { return m; }

// The identity but for three entries.
M drift(double ds) {
  M R = M::Identity();
  R(1, 2) = ds;
  R(3, 4) = ds;
  R(5, 6) = ds;
  return keep(R);
}

// Its pullback runs the products again, and they treat R as all varying. One
// remark for the template, whichever instantiations the gradient runs.
template <typename T>
void push(M& cm, T ds) { // expected-remark-re {{the pullback of 'push' runs its forward work again, about {{[0-9]+}}% of the work of the gradient of 'track'}} expected-note {{it runs 'mul', 't', 'drift' again}} expected-note {{a hand-written 'push_reverse_forw' that hands its pullback what it needs through clad::pullback_state runs it once}} expected-note {{a hand-written pullback of 'push' that touches only those entries skips the rest}}
  if (!ds)
    return;
  M R = drift(ds); // expected-remark-re {{'R' is constant except 3 of its 36 entries, which its derivative treats as all varying, about {{[0-9]+}}% of the work of the gradient of 'track'}}
  M t = mul(mul(R, cm), R.t());
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      cm.a[i][j] = t.a[i][j];
}

double track(double ds) {
  M cm = M::Identity();
  for (int i = 0; i < 8; ++i)
    push(cm, ds);
  for (int i = 0; i < 8; ++i)
    push(cm, float(ds));
  return cm(1, 1) + cm(3, 3);
}

// An object a call may change anywhere is not constant except for a few
// entries.
void scale(M& m, double k) {
  for (int i = 0; i < 6; ++i)
    m.a[i][i] *= k;
}

double whole(double k, double c) {
  M R = M::Identity();
  scale(R, k);
  return R(1, 1) * c;
}

double wholes(double k, double c) {
  double s = 0;
  for (int i = 0; i < 8; ++i)
    s += whole(k, c);
  return s;
}

// Nothing to say about a pullback that computes no more than its partials.
double sq(double x) { return x * x; }

double sum(const double* x) {
  double s = 0;
  for (int i = 0; i < 100000; ++i)
    s += sq(x[i]);
  return s;
}

int count(const M& m) { return m(1, 1) != 0; }

// An object read at constant positions only is no dense use. Its pullback still
// builds it all again for two partials.
double read(double k, double c) { // expected-remark-re {{the pullback of 'read' runs its forward work again, about {{[0-9]+}}% of the work of the gradient of 'reads'}} expected-note {{it runs 'Identity', 'operator()', 'count' again}} expected-note {{a hand-written 'read_reverse_forw' that hands its pullback what it needs through clad::pullback_state runs it once}}
  M R = M::Identity();
  R(1, 2) = k;
  return R(1, 2) * c + count(R);
}

double reads(double k, double c) {
  double s = 0;
  for (int i = 0; i < 100000; ++i)
    s += read(k, c);
  return s;
}

double corners(const M& m) { return m(1, 1) * m(6, 6) + m(1, 6) * m(6, 1); }

// Written at a position that is not a constant, R may differ anywhere. Its
// pullback still builds it again.
double anywhere(double k, int j) { // expected-remark-re {{the pullback of 'anywhere' runs its forward work again, about {{[0-9]+}}% of the work of the gradient of 'anywheres'}} expected-note {{it runs 'Identity', 'corners', 'operator()' again}} expected-note {{a hand-written 'anywhere_reverse_forw' that hands its pullback what it needs through clad::pullback_state runs it once}}
  M R = M::Identity();
  R(1, 2) = k;
  R(j, 1) = k;
  return corners(R);
}

double anywheres(double k) {
  double s = 0;
  for (int i = 0; i < 100000; ++i)
    s += anywhere(k, i % 6 + 1);
  return s;
}

// Neither is constant but for a few entries: q is assigned whole, and p
// holds a pointer.
struct N9 {
  double a, b, c, d, e, f, g, h, i;
};
const N9 Z{};
struct P9 {
  double a[9];
  double* p;
};
namespace ns {
double twice(double x) { return 2 * x; }
} // namespace ns

void reset(M& out, double k, int n) {
  N9 q = Z;
  q.a = k;
  q = Z;
  P9 p{};
  // Loops whose trip counts are not constants are taken to run a few times.
  for (int i = 0; i < n; ++i)
    k += k;
  for (int i = 0; i != 2; ++i)
    k *= k;
  int i = 0;
  while (i < 2) {
    k += 1;
    ++i;
  }
  for (int i = 0; i < 4; ++i) {
    if (k > 100)
      break;
    k += 1;
  }
  // Counted down to zero, as a reverse sweep counts.
  int c, j = 0;
  for (c = 2, ++j; c; --c)
    k += j;
  out(1, 1) = ns::twice(k) + count(out) + q.a + p.a[0];
  if (k == 0)
    return;
  out(2, 2) = k;
}

struct Holder {
  M m;
  double get(double k) const {
    M R = m;
    R(1, 2) = k;
    return R(1, 2);
  }
};

double small(double k) {
  M out{};
  reset(out, k, 3);
  Holder h{};
  return out(1, 1) + h.get(k);
}

// Nothing to say about a gradient that does no work.
double constant(double) { return 1; }

void gradients() {
  clad::gradient(track, "ds");
  clad::gradient(sum);
  clad::gradient(reads);
  clad::gradient(wholes);
  clad::gradient(anywheres);
  clad::gradient(small);
  clad::gradient(constant);
}
