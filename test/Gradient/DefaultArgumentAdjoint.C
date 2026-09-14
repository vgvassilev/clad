// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t 2>&1 | %filecheck %s
// RUN: %t
// RUN: %cladclang %s -I%S/../../include -Xclang -verify -Xclang -plugin-arg-clad -Xclang -enable-va -o %t
// RUN: %t
// RUN: %cladclang %s -I%S/../../include -Xclang -verify -Xclang -plugin-arg-clad -Xclang -disable-tbr -o %t
// RUN: %t

#include "clad/Differentiator/Differentiator.h"

struct DefaultFactor {
  struct Construct {};

  DefaultFactor() = default;
  explicit constexpr DefaultFactor(Construct) {}
};

constexpr DefaultFactor defaultFactor{DefaultFactor::Construct{}};

struct OptionalFactor {
  OptionalFactor() = default;
  OptionalFactor(DefaultFactor, int = 0) {}

  int value_or(int fallback) const { return fallback; }
};

double configured_scale(double value, OptionalFactor factor = defaultFactor) {
  return value * factor.value_or(2);
}

namespace clad::custom_derivatives {

void configured_scale_pullback(
    double /*value*/, OptionalFactor factor, double d_output, double* d_value,
    OptionalFactor* /*d_factor*/) {
  *d_value += factor.value_or(2) * d_output;
}

} // namespace clad::custom_derivatives

double use_default_factor(double value) { return configured_scale(value); }

double global_default = 2.0; // expected-warning {{gradient uses a global variable 'global_default'}}

double scale_with_global_default(double value, double factor = global_default) {
  return value * factor;
}

double use_global_default(double value) {
  global_default = value;
  return scale_with_global_default(value);
}

// CHECK-LABEL: void use_default_factor_grad(double value, double *_d_value) {
// CHECK: OptionalFactor _r1 = {{[{]}}{}, 0};
// CHECK: clad::custom_derivatives::configured_scale_pullback(value, {defaultFactor, 0}, 1, &_r0, &_r1);
// CHECK: *_d_value += _r0;
// CHECK-LABEL: void use_global_default_grad(double value, double *_d_value) {
// CHECK: scale_with_global_default_pullback(value, global_default, 1, &_r0, &_r1);
// CHECK: *_d_value += _r0;
// CHECK: _d_global_default += _r1;
// CHECK: *_d_value += _d_global_default;

namespace temporary {
struct A {
  A() {}
};
double f1(double x, A a = A()) { return 2 * x; }
double use_default_temporary(double x) { return f1(x); }
} // namespace temporary

bool check_defaults() {
  auto gradient = clad::gradient(use_default_factor);
  double derivative = 0.0;
  gradient.execute(3.0, &derivative);

  auto globalGradient = clad::gradient(use_global_default);
  double globalDerivative = 0.0;
  globalGradient.execute(3.0, &globalDerivative);

  auto temporaryGradient = clad::gradient(temporary::use_default_temporary);
  double temporaryDerivative = 0.0;
  temporaryGradient.execute(3.0, &temporaryDerivative);
  return derivative == 2.0 && globalDerivative == 6.0 &&
         temporaryDerivative == 2.0;
}

namespace scalar {
float b = 2; // expected-warning {{gradient uses a global variable 'b'}}
double f1(double a = b) { return 3 * a; }
double f2(double x) {
  b = x;
  return f1();
}
double f3(double x) {
  b = x;
  return f1(b);
}

// CHECK-LABEL: void f2_grad(double x, double *_d_x) {
// CHECK: _d_b += _r0;
// CHECK: *_d_x += scalar::_d_b;

bool check() {
  auto d2 = clad::gradient(f2);
  auto d3 = clad::gradient(f3);
  double r2 = 0, r3 = 0;
  d2.execute(2, &r2);
  d3.execute(2, &r3);
  return r2 == 3 && r3 == 3;
}
} // namespace scalar

namespace record {
struct B {
  double b;
};
B b{2}; // expected-warning {{gradient uses a global variable 'b'}}
struct A {
  double a;
  A(const B& x, int k = 3) : a(k * x.b) {}
};
double f1(A a = b) { return a.a; }
double f2(double x) {
  b.b = x;
  return f1();
}
double f3(double x) {
  b.b = x;
  return f1(b);
}

// CHECK-LABEL: void f2_grad(double x, double *_d_x) {
// CHECK: record::A::constructor_reverse_forw(clad::Tag<record::A>(), b, 3, record::_d_b, 0);
// CHECK: A::constructor_pullback(b, 3, &_r0, &record::_d_b, &_r1);
// CHECK: *_d_x += _r_d0;

bool check() {
  auto d2 = clad::gradient(f2);
  auto d3 = clad::gradient(f3);
  double r2 = 0, r3 = 0;
  d2.execute(2, &r2);
  d3.execute(2, &r3);
  return r2 == 3 && r3 == 3;
}
} // namespace record

namespace effects {
double f1(double a, OptionalFactor factor = (global_default *= 2, defaultFactor)) {
  return a;
}
double f2(double x) {
  global_default = x;
  double a = f1(x);
  return a + global_default;
}

// CHECK-LABEL: void f2_grad(double x, double *_d_x) {
// CHECK: _d_global_default += 1;
// CHECK: effects::f1_pullback
// CHECK: _d_global_default += _r_d0 * 2;
// CHECK: *_d_x += _d_global_default;

struct A {
  double a;
  A(double x) : a(x) { global_default *= 2; }
};
double f3(A a = global_default) { return a.a; }
double f4(double x) {
  global_default = x;
  double a = f3();
  return a + global_default;
}

// CHECK-LABEL: void f4_grad(double x, double *_d_x) {
// CHECK: A::constructor_pullback
// CHECK: *_d_x += _d_global_default;

bool check() {
  auto d2 = clad::gradient(f2);
  auto d4 = clad::gradient(f4);
  double r2 = 0, r4 = 0;
  d2.execute(2, &r2);
  d4.execute(2, &r4);
  return r2 == 3 && r4 == 3;
}
} // namespace effects

int main() {
  return !check_defaults() || !scalar::check() || !record::check() ||
         !effects::check();
}
