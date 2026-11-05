// RUN: %cladclang %s -I%S/../../include -fsyntax-only -ferror-limit=0 -Xclang -verify
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

double mutable_forward(double x) {
  double c = 2.0;
  auto g = [c](double t) mutable { c += 1; return t * c; }; // expected-error {{differentiation of a mutable lambda is not supported: its captures change between calls}}
  return g(x);
}

double mutable_reverse(double x) {
  double c = 2.0;
  auto g = [c](double t) mutable { c += 1; return t * c; }; // expected-error {{differentiation of a mutable lambda is not supported: its captures change between calls}}
  return g(x);
}

double byref_write_forward(double x) {
  double c = 2.0;
  auto g = [&c](double t) { c = c * t; return c; }; // expected-error {{forward-mode differentiation of a lambda that assigns to a by-reference capture is not supported yet}}
  return g(x);
}

// The closure would be rebuilt every iteration, with that iteration's captures.
double in_loop_forward(double x) {
  double s = 0;
  for (int i = 0; i < 3; ++i) {
    auto g = [x, i](double t) { return x * t * i; }; // expected-error {{differentiation of a lambda with captures created inside a loop is not supported yet}}
    s += g(x);
  }
  return s;
}

double in_loop_reverse(double x) {
  double s = 0;
  for (int i = 0; i < 3; ++i) {
    auto g = [x, i](double t) { return x * t * i; }; // expected-error {{differentiation of a lambda with captures created inside a loop is not supported yet}}
    s += g(x);
  }
  return s;
}

struct Scaled {
  double k = 3;
  double forward(double x) {
    auto f = [this](double t) { return k * t; }; // expected-error {{differentiation of a lambda that captures 'this' is not supported yet}}
    return f(x);
  }
  double reverse(double x) {
    auto f = [this](double t) { return k * t; }; // expected-error {{differentiation of a lambda that captures 'this' is not supported yet}}
    return f(x);
  }
};

double vector_mode(double x, double y) {
  auto f = [y](double t) { return t * t * y; }; // expected-error {{vector-mode differentiation of a lambda with captures is not supported yet}}
  return f(x);
}

// A lambda passed to a function template is differentiated with the calls it
// makes to it, which these lambdas do not allow.
template <class F> void apply3(const F& f) {
  for (int i = 0; i < 3; ++i)
    f(i);
}
template <class F> void applyReal(const F& f) { f(1.5); }
struct WithReal {
  double d;
};
struct Index : WithReal {
  int i;
};
template <class F> void applyIndex(const F& f) { f(Index{}); }
const int Count = 3;
template <const int* N, class F> void applyN(const F& f) { f(*N); }

double callback_generic(double x) {
  double s = 0;
  apply3([&](auto i) { s += x * i; }); // expected-error {{differentiation of a call to 'apply3' that passes a lambda with an 'auto' parameter is not supported yet}}
  return s;
}

double callback_returns(double x) {
  double s = 0;
  apply3([&](int i) { if (i == 1) return; s += x * i; }); // expected-error {{differentiation of a call to 'apply3' that passes a lambda that returns early is not supported yet}}
  return s;
}

double callback_real_param(double x) {
  double s = 0;
  applyReal([&](double t) { s += x * t; }); // expected-error {{differentiation of a call to 'applyReal' that passes a lambda whose parameter is not an integer is not supported yet}}
  return s;
}

double callback_nested(double x) {
  double s = 0;
  apply3([&](int i) { apply3([&](int j) { s += x * i * j; }); }); // expected-error {{differentiation of a call to 'apply3' that passes a lambda that itself passes a lambda to a function template is not supported yet}}
  return s;
}

double callback_value(double x) {
  double s = 0;
  apply3([&](int i) { s += x * i; return i; }); // expected-error {{differentiation of a call to 'apply3' that passes a lambda that returns a value is not supported yet}}
  return s;
}

double callback_init_capture(double x) {
  double s = 0;
  apply3([&, k = 2](int i) { s += x * i * k; }); // expected-error {{differentiation of a call to 'apply3' that passes a lambda that captures 'this' or an initializer is not supported yet}}
  return s;
}

double callback_base_real(double x) {
  double s = 0;
  applyIndex([&](Index k) { s += x * k.i; }); // expected-error {{differentiation of a call to 'applyIndex' that passes a lambda whose parameter is not an integer is not supported yet}}
  return s;
}

double callback_pointer_arg(double x) {
  double s = 0;
  applyN<&Count>([&](int i) { s += x * i; }); // expected-error {{differentiation of a call to 'applyN' that passes a lambda to a function with this kind of template argument is not supported yet}}
  return s;
}

// Only the lambda's body is differentiated, so the function it is passed to
// may not touch a floating-point value of its own, directly or through a call.
double Scale = 1;
template <class F> void scaleThen(const F& f) {
  Scale *= 3;
  f(0);
}
template <class F> void readThen(const F& f) { f(Scale > 1); }
double tick();
template <class F> void tickThen(const F& f) {
  tick();
  f(0);
}

// A constructor or a destructor the function runs counts as a call.
struct Tally {
  Tally() { Scale += 1; }
};
struct InitTally {
  int v;
  InitTally() : v(int(Scale += 1)) {}
};
struct Untally {
  ~Untally() { Scale -= 1; }
};
template <class F> void tallyThen(const F& f) {
  Tally t;
  f(0);
}
template <class F> void untallyThen(const F& f) {
  Untally u;
  f(0);
}
template <class F> void initTallyThen(const F& f) {
  InitTally t;
  f(0);
}
void tock();
template <class F> void tockThen(const F& f) {
  tock();
  f(0);
}

double callback_constructor(double x) {
  double s = 0;
  tallyThen([&](int i) { s += x * i; }); // expected-error {{differentiation of a call to 'tallyThen' that passes a lambda to a function that does more than call it is not supported yet}}
  return s;
}

double callback_destructor(double x) {
  double s = 0;
  untallyThen([&](int i) { s += x * i; }); // expected-error {{differentiation of a call to 'untallyThen' that passes a lambda to a function that does more than call it is not supported yet}}
  return s;
}

double callback_initializer(double x) {
  double s = 0;
  initTallyThen([&](int i) { s += x * i; }); // expected-error {{differentiation of a call to 'initTallyThen' that passes a lambda to a function that does more than call it is not supported yet}}
  return s;
}

double callback_no_body(double x) {
  double s = 0;
  tockThen([&](int i) { s += x * i; }); // expected-error {{differentiation of a call to 'tockThen' that passes a lambda to a function that does more than call it is not supported yet}}
  return s;
}

double callback_writes_global(double x) {
  double s = 0;
  scaleThen([&](int i) { s += x * i; }); // expected-error {{differentiation of a call to 'scaleThen' that passes a lambda to a function that does more than call it is not supported yet}}
  return s;
}

double callback_reads_global(double x) {
  double s = 0;
  readThen([&](int i) { s += x * i; }); // expected-error {{differentiation of a call to 'readThen' that passes a lambda to a function that does more than call it is not supported yet}}
  return s;
}

double callback_calls_other(double x) {
  double s = 0;
  tickThen([&](int i) { s += x * i; }); // expected-error {{differentiation of a call to 'tickThen' that passes a lambda to a function that does more than call it is not supported yet}}
  return s;
}

double callback_forward(double x) {
  double s = 0;
  apply3([&](auto i) { s += x * i; }); // expected-error {{differentiation of a call to 'apply3' that passes a lambda with an 'auto' parameter is not supported yet}}
  return s;
}

int main() {
  clad::gradient(callback_generic);
  clad::gradient(callback_returns);
  clad::gradient(callback_real_param);
  clad::gradient(callback_nested);
  clad::differentiate(callback_forward, 0);
  clad::gradient(callback_value);
  clad::gradient(callback_init_capture);
  clad::gradient(callback_base_real);
  clad::gradient(callback_pointer_arg);
  clad::gradient(callback_writes_global);
  clad::gradient(callback_reads_global);
  clad::gradient(callback_calls_other);
  clad::gradient(callback_constructor);
  clad::gradient(callback_destructor);
  clad::gradient(callback_initializer);
  clad::gradient(callback_no_body);
  clad::differentiate(&Scaled::forward, "x");
  clad::gradient(&Scaled::reverse);
  clad::differentiate<clad::opts::vector_mode>(vector_mode, "x,y");
  clad::differentiate(mutable_forward, 0);
  clad::differentiate(in_loop_forward, 0);
  clad::gradient(in_loop_reverse);
  clad::gradient(mutable_reverse);
  clad::differentiate(byref_write_forward, 0);
}
