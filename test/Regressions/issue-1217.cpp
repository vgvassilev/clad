// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck %s --check-prefix=CHECK-EXEC
// RUN: %cladclang %s -I%S/../../include -Xclang -verify -Xclang -plugin-arg-clad -Xclang -disable-tbr -o %t.notbr
// RUN: %t.notbr | %filecheck %s --check-prefix=CHECK-EXEC
// RUN: %cladclang %s -I%S/../../include -DDIAGNOSTICS -fsyntax-only -Xclang -verify
// RUN: %cladclang %s -I%S/../../include -DDIAGNOSTICS -fsyntax-only -Xclang -verify -Xclang -plugin-arg-clad -Xclang -disable-tbr

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct A {
  double x;
  A operator=(const A&) = delete;
};

struct M {
  double x;
  M& operator=(const M&) = delete;
  M& operator=(M&&) = default;
};

struct Copyable {
  double x;
};

#ifndef DIAGNOSTICS
// expected-no-diagnostics

double issue(double x, double y) {
  while (x) {
    A u;
  }
  return x;
}

double linear_block(double x) {
  double r = 0;
  if (x > 0) {
    A u{x};
    r = u.x * 2;
  }
  return r;
}

double linear_loop(double x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    A u{x * i};
    r += u.x * 2;
  }
  return r;
}

double mixed(double x) {
  if (x > 0) {
    A u{x}, *p = nullptr;
    x = u.x * 2;
  }
  return x;
}

double movable(double x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    M m{x * i};
    r += m.x;
  }
  return r;
}

double copyable(double x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    Copyable u{x * i};
    r += u.x * x;
  }
  return r;
}

int main() {
  auto d_issue = clad::gradient(issue);
  double dx = 0, dy = 0;
  d_issue.execute(0, 1, &dx, &dy);
  printf("issue: %.1f %.1f\n", dx, dy); // CHECK-EXEC: issue: 1.0 0.0
  auto d_block = clad::gradient(linear_block);
  dx = 0;
  d_block.execute(3, &dx);
  printf("block: %.1f\n", dx); // CHECK-EXEC-NEXT: block: 2.0
  auto d_loop = clad::gradient(linear_loop, "x");
  dx = 0;
  d_loop.execute(3, 3, &dx);
  printf("loop: %.1f\n", dx); // CHECK-EXEC-NEXT: loop: 6.0
  auto d_mixed = clad::gradient(mixed);
  dx = 0;
  d_mixed.execute(3, &dx);
  printf("mixed: %.1f\n", dx); // CHECK-EXEC-NEXT: mixed: 2.0
  auto d_movable = clad::gradient(movable, "x");
  dx = 0;
  d_movable.execute(3, 3, &dx);
  printf("movable: %.1f\n", dx); // CHECK-EXEC-NEXT: movable: 3.0
  auto d_copyable = clad::gradient(copyable, "x");
  dx = 0;
  d_copyable.execute(3, 3, &dx);
  printf("copyable: %.1f\n", dx); // CHECK-EXEC-NEXT: copyable: 18.0
}

#else
double nonlinear(double x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    A u{x * i}; // expected-error {{cannot differentiate local variable 'u' of type 'A': reverse storage requires unavailable assignment from an rvalue}}
    r += u.x * x;
  }
  return r;
}

double nonlinear_movable(double x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    M m{x * i}; // expected-error {{cannot differentiate local variable 'm' of type 'M': reverse storage requires unavailable copy construction}}
    r += m.x * x;
  }
  return r;
}

double address_escape(double x) {
  double r = 0;
  if (x > 0) {
    A u{x}; // expected-error {{cannot differentiate local variable 'u' of type 'A': reverse storage requires unavailable assignment from an rvalue}}
    A* p = &u;
    r = p->x * 2;
  }
  return r;
}

double reference_escape(double x) {
  double r = 0;
  if (x > 0) {
    A u{x}; // expected-error {{cannot differentiate local variable 'u' of type 'A': reverse storage requires unavailable assignment from an rvalue}}
    A& ref = u;
    r = ref.x * 2;
  }
  return r;
}

struct Nonaggregate {
  double x;
  Nonaggregate(double v) : x(v) {}
  Nonaggregate& operator=(const Nonaggregate&) = delete;
};

double nonaggregate(double x) {
  if (x > 0) {
    Nonaggregate u{x}; // expected-error {{cannot differentiate local variable 'u' of type 'Nonaggregate': non-aggregate storage requires unavailable assignment from an rvalue}}
  }
  return x;
}

double range(double x) {
  A values[1] = {{x}};
  for (A u : values) { // expected-error {{cannot differentiate local variable 'u' of type 'A': range-for storage requires unavailable assignment from an rvalue}}
  }
  return x;
}

int main() {
  clad::gradient(nonlinear, "x");
  clad::gradient(nonlinear_movable, "x");
  clad::gradient(address_escape);
  clad::gradient(reference_escape);
  clad::gradient(nonaggregate);
  clad::gradient(range);
}
#endif

// CHECK-LABEL: void issue_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT: A _d_u = {0.};
// CHECK-NEXT: unsigned {{int|long|long long}} _t0 = 0;
// CHECK-NEXT: while (x)
// CHECK-NEXT: {
// CHECK-NEXT: _t0++;
// CHECK-NEXT: A u;
// CHECK-NEXT: }
// CHECK-NEXT: *_d_x += 1;
// CHECK-NEXT: while (_t0)
// CHECK-NEXT: {
// CHECK-NEXT: {
// CHECK-NEXT: clad::zero_init(_d_u);
// CHECK-NEXT: }
// CHECK-NEXT: _t0--;
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-LABEL: void linear_block_grad(double x, double *_d_x) {
// CHECK-NEXT: bool _cond0;
// CHECK-NEXT: A _d_u = {0.};
// CHECK-NEXT: double _d_r = 0.;
// CHECK-NEXT: double r = 0;
// CHECK-NEXT: {
// CHECK-NEXT: _cond0 = x > 0;
// CHECK-NEXT: if (_cond0) {
// CHECK-NEXT: A u{x};
// CHECK-NEXT: r = u.x * 2;
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: _d_r += 1;
// CHECK-NEXT: if (_cond0) {
// CHECK-NEXT: {
// CHECK-NEXT: double _r_d0 = _d_r;
// CHECK-NEXT: _d_r = 0.;
// CHECK-NEXT: _d_u.x += _r_d0 * 2;
// CHECK-NEXT: }
// CHECK-NEXT: *_d_x += _d_u.x;
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-LABEL: void linear_loop_grad_0(double x, int n, double *_d_x) {
// CHECK-NEXT: int _d_n = 0;
// CHECK-NEXT: int _d_i = 0;
// CHECK-NEXT: int i = 0;
// CHECK-NEXT: A _d_u = {0.};
// CHECK-NEXT: double _d_r = 0.;
// CHECK-NEXT: double r = 0;
// CHECK-NEXT: unsigned {{int|long|long long}} _t0;
// CHECK-NEXT: for (i = 0; i < n; ++i) {
// CHECK-NEXT: A u{x * i};
// CHECK-NEXT: r += u.x * 2;
// CHECK-NEXT: }
// CHECK-NEXT: _d_r += 1;
// CHECK-NEXT: for (_t0 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-NEXT: --i;
// CHECK-NEXT: {
// CHECK-NEXT: double _r_d0 = _d_r;
// CHECK-NEXT: _d_u.x += _r_d0 * 2;
// CHECK-NEXT: }
// CHECK-NEXT: {
// CHECK-NEXT: *_d_x += _d_u.x * i;
// CHECK-NEXT: _d_i += x * _d_u.x;
// CHECK-NEXT: clad::zero_init(_d_u);
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-LABEL: void mixed_grad(double x, double *_d_x) {
// CHECK-NEXT: bool _cond0;
// CHECK-NEXT: A _d_u = {0.}, *_d_p = nullptr;
// CHECK-NEXT: A *p = nullptr;
// CHECK-NEXT: {
// CHECK-NEXT: _cond0 = x > 0;
// CHECK-NEXT: if (_cond0) {
// CHECK-NEXT: _d_p = nullptr;
// CHECK-NEXT: A u{x};
// CHECK-NEXT: p = nullptr;
// CHECK-NEXT: x = u.x * 2;
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: *_d_x += 1;
// CHECK-NEXT: if (_cond0) {
// CHECK-NEXT: {
// CHECK-NEXT: double _r_d0 = *_d_x;
// CHECK-NEXT: *_d_x = 0.;
// CHECK-NEXT: _d_u.x += _r_d0 * 2;
// CHECK-NEXT: }
// CHECK-NEXT: *_d_x += _d_u.x;
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-LABEL: void movable_grad_0(double x, int n, double *_d_x) {
// CHECK-NEXT: int _d_n = 0;
// CHECK-NEXT: int _d_i = 0;
// CHECK-NEXT: int i = 0;
// CHECK-NEXT: M _d_m = {0.};
// CHECK-NEXT: M m = {0.};
// CHECK-NEXT: double _d_r = 0.;
// CHECK-NEXT: double r = 0;
// CHECK-NEXT: unsigned {{int|long|long long}} _t0;
// CHECK-NEXT: for (i = 0; i < n; ++i) {
// CHECK-NEXT: m = {x * i};
// CHECK-NEXT: r += m.x;
// CHECK-NEXT: }
// CHECK-NEXT: _d_r += 1;
// CHECK-NEXT: for (_t0 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-NEXT: --i;
// CHECK-NEXT: {
// CHECK-NEXT: double _r_d0 = _d_r;
// CHECK-NEXT: _d_m.x += _r_d0;
// CHECK-NEXT: }
// CHECK-NEXT: {
// CHECK-NEXT: *_d_x += _d_m.x * i;
// CHECK-NEXT: _d_i += x * _d_m.x;
// CHECK-NEXT: _d_m = {0.};
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-LABEL: void copyable_grad_0(double x, int n, double *_d_x) {
// CHECK-NEXT: int _d_n = 0;
// CHECK-NEXT: int _d_i = 0;
// CHECK-NEXT: int i = 0;
// CHECK-NEXT: clad::tape<Copyable> _t1 = {};
// CHECK-NEXT: Copyable _d_u = {0.};
// CHECK-NEXT: Copyable u = {0.};
// CHECK-NEXT: double _d_r = 0.;
// CHECK-NEXT: double r = 0;
// CHECK-NEXT: unsigned {{int|long|long long}} _t0;
// CHECK-NEXT: for (i = 0; i < n; ++i) {
// CHECK-NEXT: clad::push(_t1, std::move(u)) , u = {x * i};
// CHECK-NEXT: r += u.x * x;
// CHECK-NEXT: }
// CHECK-NEXT: _d_r += 1;
// CHECK-NEXT: for (_t0 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-NEXT: --i;
// CHECK-NEXT: {
// CHECK-NEXT: double _r_d0 = _d_r;
// CHECK-NEXT: _d_u.x += _r_d0 * x;
// CHECK-NEXT: *_d_x += u.x * _r_d0;
// CHECK-NEXT: }
// CHECK-NEXT: {
// CHECK-NEXT: *_d_x += _d_u.x * i;
// CHECK-NEXT: _d_i += x * _d_u.x;
// CHECK-NEXT: _d_u = {0.};
// CHECK-NEXT: u = clad::pop(_t1);
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
