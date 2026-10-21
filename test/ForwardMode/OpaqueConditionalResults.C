// RUN: %cladclang %s -I%S/../../include -o %t -Xclang -verify 2>&1 | %filecheck %s
// RUN: %t | %filecheck %s --check-prefix=EXEC
// expected-no-diagnostics

// Results that an opaque call leaves its surroundings to rebuild: void, lvalue,
// xvalue and class conditionals, and operands that reach the evaluation path
// wrapped in something other than a cast, a conditional or a comma.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

int calls = 0;
double bump(double x) { ++calls; return x * x; }
void scale(double& v) { v *= 2; }
double&& pass(double& v) { return static_cast<double&&>(v); }
CLAD_NONDIFFERENTIABLE double opaque(double value) { return value; }
CLAD_NONDIFFERENTIABLE void opaque_void(double) {}
CLAD_NONDIFFERENTIABLE double take_rvalue(double&& value) { return value; }
CLAD_NONDIFFERENTIABLE double opaque_default(double v,
                                             double w = (++calls, 0.)) {
  return v + w;
}
struct P { double v; };
CLAD_NONDIFFERENTIABLE P opaque_p(double v) { return {v}; }

// Neither branch has a tangent.
double void_without_tangent(double x, bool enabled) {
  double r = x;
  enabled ? opaque_void(x) : scale(r);
  return r;
}

// CHECK: double void_without_tangent_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = _d_x;
// CHECK-NEXT:     double r = x;
// CHECK-NEXT:     enabled ? opaque_void(x) : scale_pushforward(r, _d_r);
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }

// One branch has statements of its own.
double void_with_prelude(double x, bool enabled) {
  double r = 0;
  enabled ? (void)(r = bump(x)) : opaque_void(x);
  return r;
}

// CHECK: double void_with_prelude_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     enabled ? (void)[&]() -> double & {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t2 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         return (_d_r = _t2.pushforward) , (r = _t2.value);
// CHECK-NEXT:     }() : opaque_void(x);
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }

// A write through the result reaches the tangent of the selected variable.
double lvalue_result(double x, bool enabled) {
  double r = 0, s = x;
  double& ref = enabled ? (opaque(x), r = bump(x)) : s;
  ref *= 2;
  return r + s;
}

// CHECK: double lvalue_result_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0, _d_s = _d_x;
// CHECK-NEXT:     double r = 0, s = x;
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _cond0 = enabled ? [&]() -> clad::ValueAndPushforward<double &, double &> {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         double &_d_cond = (opaque(x) , (_d_r = _t1.pushforward));
// CHECK-NEXT:         return {(r = _t1.value), _d_cond};
// CHECK-NEXT:     }() : [&]() -> clad::ValueAndPushforward<double &, double &> {
// CHECK-NEXT:         double &_d_cond = _d_s;
// CHECK-NEXT:         return {s, _d_cond};
// CHECK-NEXT:     }();
// CHECK-NEXT:     double &_d_ref = _cond0.pushforward;
// CHECK-NEXT:     double &ref = _cond0.value;
// CHECK-NEXT:     _d_ref = _d_ref * 2;
// CHECK-NEXT:     ref *= 2;
// CHECK-NEXT:     return _d_r + _d_s;
// CHECK-NEXT: }

double xvalue_result(double x, bool enabled) {
  double r = 0, s = x;
  double&& ref = enabled ? static_cast<double&&>((opaque(x), r = bump(x)))
                         : static_cast<double&&>(s);
  ref *= 2;
  return r + s;
}

// CHECK: double xvalue_result_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0, _d_s = _d_x;
// CHECK-NEXT:     double r = 0, s = x;
// CHECK-NEXT:     clad::ValueAndPushforward<double &&, double &&> _cond0 = enabled ? [&]() -> clad::ValueAndPushforward<double &&, double &&> {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         double &&_d_cond = static_cast<double &&>((opaque(x) , (_d_r = _t1.pushforward)));
// CHECK-NEXT:         return {static_cast<double &&>((r = _t1.value)), static_cast<double &&>(_d_cond)};
// CHECK-NEXT:     }() : [&]() -> clad::ValueAndPushforward<double &&, double &&> {
// CHECK-NEXT:         double &&_d_cond = static_cast<double &&>(_d_s);
// CHECK-NEXT:         return {static_cast<double &&>(s), static_cast<double &&>(_d_cond)};
// CHECK-NEXT:     }();
// CHECK-NEXT:     double &&_d_ref = static_cast<double &&>(_cond0.pushforward);
// CHECK-NEXT:     double &&ref = static_cast<double &&>(_cond0.value);
// CHECK-NEXT:     _d_ref = _d_ref * 2;
// CHECK-NEXT:     ref *= 2;
// CHECK-NEXT:     return _d_r + _d_s;
// CHECK-NEXT: }

// A temporary in one branch survives the lambda; the other branch still aliases s.
double temporary_result(double x, bool enabled) {
  double s = x;
  double&& ref = enabled ? (opaque(x), static_cast<double&&>(bump(x)))
                         : static_cast<double&&>(s);
  ref *= 2;
  return ref + s;
}

// CHECK: double temporary_result_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _cond_value0 = 0.;
// CHECK-NEXT:     double _cond_tangent0 = 0.;
// CHECK-NEXT:     double _d_s = _d_x;
// CHECK-NEXT:     double s = x;
// CHECK-NEXT:     clad::ValueAndPushforward<double &&, double &&> _cond0 = enabled ? [&]() -> clad::ValueAndPushforward<double &&, double &&> {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         double &&_d_cond = static_cast<double &&>(_cond_tangent0 = (opaque(x) , static_cast<double &&>(_t1.pushforward)));
// CHECK-NEXT:         _cond_value0 = static_cast<double &&>(_t1.value);
// CHECK-NEXT:         return {static_cast<double &&>(_cond_value0), static_cast<double &&>(_d_cond)};
// CHECK-NEXT:     }() : [&]() -> clad::ValueAndPushforward<double &&, double &&> {
// CHECK-NEXT:         double &&_d_cond = static_cast<double &&>(_d_s);
// CHECK-NEXT:         return {static_cast<double &&>(s), static_cast<double &&>(_d_cond)};
// CHECK-NEXT:     }();
// CHECK-NEXT:     double &&_d_ref = static_cast<double &&>(_cond0.pushforward);
// CHECK-NEXT:     double &&ref = static_cast<double &&>(_cond0.value);
// CHECK-NEXT:     _d_ref = _d_ref * 2;
// CHECK-NEXT:     ref *= 2;
// CHECK-NEXT:     return _d_ref + _d_s;
// CHECK-NEXT: }


double nested_temporary_result(double x, bool enabled, bool temporary) {
  double s = x;
  double&& ref = enabled
                     ? (opaque(x), temporary ? static_cast<double&&>(bump(x))
                                           : static_cast<double&&>(s))
                     : static_cast<double&&>(s);
  ref *= 2;
  return ref + s;
}

// CHECK: double nested_temporary_result_darg0(double x, bool enabled, bool temporary) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     bool _d_temporary = 0;
// CHECK-NEXT:     double _cond_value0 = 0.;
// CHECK-NEXT:     double _cond_tangent0 = 0.;
// CHECK-NEXT:     double _d_s = _d_x;
// CHECK-NEXT:     double s = x;
// CHECK-NEXT:     clad::ValueAndPushforward<double &&, double &&> _cond1 = enabled ? [&]() -> clad::ValueAndPushforward<double &&, double &&> {
// CHECK-NEXT:         clad::ValueAndPushforward<double &&, double &&> _cond0 = temporary ? [&]() -> clad::ValueAndPushforward<double &&, double &&> {
// CHECK-NEXT:             clad::ValueAndPushforward<double, double> _t2 = bump_pushforward(x, _d_x);
// CHECK-NEXT:             double &&_d_cond = static_cast<double &&>(_cond_tangent0 = static_cast<double &&>(_t2.pushforward));
// CHECK-NEXT:             _cond_value0 = static_cast<double &&>(_t2.value);
// CHECK-NEXT:             return {static_cast<double &&>(_cond_value0), static_cast<double &&>(_d_cond)};
// CHECK-NEXT:         }() : [&]() -> clad::ValueAndPushforward<double &&, double &&> {
// CHECK-NEXT:             double &&_d_cond = static_cast<double &&>(_d_s);
// CHECK-NEXT:             return {static_cast<double &&>(s), static_cast<double &&>(_d_cond)};
// CHECK-NEXT:         }();
// CHECK-NEXT:         double &&_d_cond = (opaque(x) , static_cast<double &&>(_cond0.pushforward));
// CHECK-NEXT:         return {static_cast<double &&>(_cond0.value), static_cast<double &&>(_d_cond)};
// CHECK-NEXT:     }() : [&]() -> clad::ValueAndPushforward<double &&, double &&> {
// CHECK-NEXT:         double &&_d_cond = static_cast<double &&>(_d_s);
// CHECK-NEXT:         return {static_cast<double &&>(s), static_cast<double &&>(_d_cond)};
// CHECK-NEXT:     }();
// CHECK-NEXT:     double &&_d_ref = static_cast<double &&>(_cond1.pushforward);
// CHECK-NEXT:     double &&ref = static_cast<double &&>(_cond1.value);
// CHECK-NEXT:     _d_ref = _d_ref * 2;
// CHECK-NEXT:     ref *= 2;
// CHECK-NEXT:     return _d_ref + _d_s;
// CHECK-NEXT: }


double const_temporary_result(double x, bool enabled) {
  double s = x;
  const double& ref = enabled ? (opaque(x), static_cast<const double&>(bump(x)))
                              : static_cast<const double&>(s);
  return ref + s;
}

// CHECK: double const_temporary_result_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _cond_value0 = 0.;
// CHECK-NEXT:     double _cond_tangent0 = 0.;
// CHECK-NEXT:     double _d_s = _d_x;
// CHECK-NEXT:     double s = x;
// CHECK-NEXT:     clad::ValueAndPushforward<const double &, const double &> _cond0 = enabled ? [&]() -> clad::ValueAndPushforward<const double &, const double &> {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         const double &_d_cond = _cond_tangent0 = (opaque(x) , static_cast<const double &>(_t1.pushforward));
// CHECK-NEXT:         _cond_value0 = static_cast<const double &>(_t1.value);
// CHECK-NEXT:         return {_cond_value0, _d_cond};
// CHECK-NEXT:     }() : [&]() -> clad::ValueAndPushforward<const double &, const double &> {
// CHECK-NEXT:         const double &_d_cond = static_cast<const double &>(_d_s);
// CHECK-NEXT:         return {static_cast<const double &>(s), _d_cond};
// CHECK-NEXT:     }();
// CHECK-NEXT:     const double &_d_ref = _cond0.pushforward;
// CHECK-NEXT:     const double &ref = _cond0.value;
// CHECK-NEXT:     return _d_ref + _d_s;
// CHECK-NEXT: }


// The opaque branch's tangent is a value of the class.
double class_result(double x, bool enabled) {
  P p = enabled ? opaque_p(x) : P{bump(x)};
  return p.v;
}

// CHECK: double class_result_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<P, P> _cond0 = enabled ? [&]() -> clad::ValueAndPushforward<P, P> {
// CHECK-NEXT:         P _d_cond = {0.};
// CHECK-NEXT:         return {opaque_p(x), _d_cond};
// CHECK-NEXT:     }() : [&]() -> clad::ValueAndPushforward<P, P> {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         P _d_cond = P({_t1.pushforward});
// CHECK-NEXT:         return {P({_t1.value}), _d_cond};
// CHECK-NEXT:     }();
// CHECK-NEXT:     P _d_p = _cond0.pushforward;
// CHECK-NEXT:     P p = _cond0.value;
// CHECK-NEXT:     return _d_p.v;
// CHECK-NEXT: }

// Without an opaque call the branches keep their statements in front of the
// conditional, as they did before.
double selected_call(double x, bool enabled) {
  return enabled ? bump(x) : x;
}

// CHECK: double selected_call_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = bump_pushforward(x, _d_x);
// CHECK-NEXT:     return enabled ? _t0.pushforward : _d_x;
// CHECK-NEXT: }

// The default argument is evaluated once.
double default_argument(double x) {
  opaque_default(x);
  return x;
}

// CHECK: double default_argument_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     opaque_default(x, ++calls , 0.);
// CHECK-NEXT:     return _d_x;
// CHECK-NEXT: }

// An xvalue operand with statements of its own.
double rvalue_operand(double x) {
  double r = 0;
  take_rvalue(pass(r = bump(x)));
  return r;
}

// CHECK: double rvalue_operand_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     take_rvalue([&]() -> double && {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t2 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         clad::ValueAndPushforward<double &&, double &&> _t3 = pass_pushforward(r = _t2.value, _d_r = _t2.pushforward);
// CHECK-NEXT:         return static_cast<double &&>(_t3.value);
// CHECK-NEXT:     }());
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }

#define CHECK_BOOL(F)                                                          \
  do {                                                                         \
    auto D = clad::differentiate(F, "x");                                      \
    for (bool enabled : {false, true}) {                                       \
      calls = 0;                                                               \
      double result = D.execute(2., enabled);                                  \
      printf(#F " %d %.1f %d\n", enabled, result, calls);                      \
    }                                                                          \
  } while (false)
#define CHECK_VALUE(F)                                                         \
  do {                                                                         \
    auto D = clad::differentiate(F, "x");                                      \
    calls = 0;                                                                 \
    double result = D.execute(2.);                                             \
    printf(#F " %.1f %d\n", result, calls);                                    \
  } while (false)

int main() {
  CHECK_BOOL(void_without_tangent);
  CHECK_BOOL(void_with_prelude);
  CHECK_BOOL(lvalue_result);
  CHECK_BOOL(xvalue_result);
  CHECK_BOOL(temporary_result);
  auto Nested = clad::differentiate(nested_temporary_result, "x");
  for (bool enabled : {false, true}) {
    for (bool temporary : {false, true}) {
      calls = 0;
      double result = Nested.execute(2., enabled, temporary);
      printf("nested_temporary_result %d %d %.1f %d\n", enabled, temporary,
             result, calls);
    }
  }
  CHECK_BOOL(const_temporary_result);
  CHECK_BOOL(class_result);
  CHECK_BOOL(selected_call);
  CHECK_VALUE(default_argument);
  CHECK_VALUE(rvalue_operand);
}
// EXEC: void_without_tangent 0 2.0 0
// EXEC-NEXT: void_without_tangent 1 1.0 0
// EXEC-NEXT: void_with_prelude 0 0.0 0
// EXEC-NEXT: void_with_prelude 1 4.0 1
// EXEC-NEXT: lvalue_result 0 2.0 0
// EXEC-NEXT: lvalue_result 1 9.0 1
// EXEC-NEXT: xvalue_result 0 2.0 0
// EXEC-NEXT: xvalue_result 1 9.0 1
// EXEC-NEXT: temporary_result 0 4.0 0
// EXEC-NEXT: temporary_result 1 9.0 1
// EXEC-NEXT: nested_temporary_result 0 0 4.0 0
// EXEC-NEXT: nested_temporary_result 0 1 4.0 0
// EXEC-NEXT: nested_temporary_result 1 0 4.0 0
// EXEC-NEXT: nested_temporary_result 1 1 9.0 1
// EXEC-NEXT: const_temporary_result 0 2.0 0
// EXEC-NEXT: const_temporary_result 1 5.0 1
// EXEC-NEXT: class_result 0 4.0 1
// EXEC-NEXT: class_result 1 0.0 0
// EXEC-NEXT: selected_call 0 1.0 1
// EXEC-NEXT: selected_call 1 4.0 1
// EXEC-NEXT: default_argument 1.0 1
// EXEC-NEXT: rvalue_operand 4.0 1
