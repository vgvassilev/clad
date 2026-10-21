// RUN: %cladclang %s -DMODE=0 -I%S/../../include -o %t.scalar -Xclang -verify 2>&1 | %filecheck %s --check-prefix=SCALAR
// RUN: %t.scalar | %filecheck %s --check-prefix=EXEC
// RUN: %cladclang %s -DMODE=1 -I%S/../../include -o %t.vector -Xclang -verify 2>&1 | %filecheck %s --check-prefix=VECTOR
// RUN: %t.vector | %filecheck %s --check-prefix=EXEC
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

int calls = 0;
double bump(double x) { ++calls; return x * x; }
CLAD_NONDIFFERENTIABLE double opaque(double value) { return value; }
CLAD_NONDIFFERENTIABLE double take_rvalue(double&& value) { return value; }
CLAD_NONDIFFERENTIABLE bool opaque_bool(double value) {
  ++calls;
  return value != 0;
}
#pragma clang diagnostic ignored "-Wunused-result"
[[nodiscard]] CLAD_NONDIFFERENTIABLE double counted(double value) {
  ++calls;
  return value;
}
struct Sink {
  CLAD_NONDIFFERENTIABLE void ignore(double) const {}
};
struct TaggedSink { void ignore(double) const {} };
CLAD_NONDIFFERENTIABLE_TYPE(TaggedSink);
struct Functor {
  CLAD_NONDIFFERENTIABLE void operator()(double) const {}
};

double argument_conditional(double x, bool enabled) {
  double r = 0;
  opaque(enabled ? r = bump(x) : 0.);
  return r;
}
double argument_logical(double x, bool enabled) {
  double r = 0;
  opaque(enabled && (r = bump(x)));
  return r;
}
double nested_conditional(double x, bool enabled) {
  double r = 0;
  opaque(enabled ? (enabled ? r = bump(x) : 0.) : 0.);
  return r;
}
double lazy_product(double x, bool enabled) {
  double r = 0;
  bool unused = enabled && (opaque(r = bump(x)) * x);
  return r;
}
double selected_product(double x, bool enabled) {
  double r = 0;
  double value = enabled ? opaque(r = bump(x)) * x : x * x;
  return r + value;
}
double if_condition(double x, bool enabled) {
  double r = 0;
  if (enabled && opaque(r = bump(x))) {}
  return r;
}
double loop_condition(double x, bool enabled) {
  double r = 0;
  int n = 0;
  while (enabled && n++ < 2 && opaque(r = bump(x))) { x += 1; }
  return r;
}
double for_variable(double x, bool enabled) {
  double r = 0;
  int n = 0;
  for (; double t = enabled && n++ < 2 ? opaque(r = bump(x)) : 0.;)
    x += 1;
  return r;
}

double while_variable(double x, bool enabled) {
  double r = 0;
  int n = 0;
  while (double t = enabled && n++ < 2 ? opaque(r = bump(x)) : 0.)
    x += 1;
  return r;
}

double do_condition(double x, bool enabled) {
  double r = 0;
  int n = 0;
  do {} while (enabled && n++ < 2 && opaque(r = bump(x)));
  return r;
}

double switch_condition(double x, bool enabled) {
  double r = 0;
  switch (enabled ? static_cast<int>(opaque(r = bump(x))) : 0) {
  default: break;
  }
  return r;
}
double receiver(double x) {
  double r = 0;
  Sink s;
  (r = x * x, s).ignore(0.);
  return r;
}
double tagged(double x) {
  double r = 0;
  TaggedSink s;
  s.ignore(r = x * x);
  return r;
}
double operator_call(double x) {
  double r = 0;
  Functor s;
  s(r = x * x);
  return r;
}
double rvalue_argument(double x) {
  double r = 0;
  take_rvalue(bump(r = x * x));
  return r;
}
double cast_rvalue_argument(double x) {
  double r = 0;
  take_rvalue(static_cast<double&&>(bump(r = x * x)));
  return r;
}
double rvalue_alias(double x) {
  double r = 0;
  take_rvalue(static_cast<double&&>(r = bump(x)));
  return r;
}
double ordinary_condition(double x) {
  double y = x;
  if (y += y) {}
  return y;
}

double cast_operands(double x) {
  double r = 0;
  take_rvalue((static_cast<double&&>(bump(r = x * x))));
  take_rvalue((double&&)(bump(r = x * x)));
  opaque(double(r = bump(x)));
  take_rvalue(const_cast<double&&>(r = bump(x)));
  take_rvalue(reinterpret_cast<double&&>(r = bump(x)));
  return r;
}

double opaque_results(double x, bool enabled) {
  double value = opaque(x);
  bool flag = enabled ? opaque_bool(x) : false;
  return x + value + (flag ? value : 0.);
}

double comma_operand(double x) {
  double r = 0;
  opaque((counted(r = bump(x)), 0.));
  return r;
}

#if MODE == 0
#define DIFF(F) clad::differentiate(F, "x")
#define BOOL_EXEC(D, X, B, R) R = D.execute(X, B)
#define EXEC(D, X, R) R = D.execute(X)
#else
#define DIFF(F) clad::differentiate<clad::opts::vector_mode>(F, "x")
#define BOOL_EXEC(D, X, B, R) D.execute(X, B, &R)
#define EXEC(D, X, R) D.execute(X, &R)
#endif
#define CHECK_BOOL(F) do { \
  auto D = DIFF(F); \
  for (bool enabled : {false, true}) { \
    calls = 0; double result = 0; BOOL_EXEC(D, 2., enabled, result); \
    printf(#F " %d %.1f %d\n", enabled, result, calls); \
  } \
} while (false)
#define CHECK_VALUE(F) do { \
  auto D = DIFF(F); \
  calls = 0; double result = 0; EXEC(D, 2., result); \
  printf(#F " %.1f %d\n", result, calls); \
} while (false)

int main() {
  CHECK_BOOL(argument_conditional);
  CHECK_BOOL(argument_logical);
  CHECK_BOOL(nested_conditional);
  CHECK_BOOL(lazy_product);
  CHECK_BOOL(selected_product);
  CHECK_BOOL(if_condition);
  CHECK_BOOL(loop_condition);
  CHECK_BOOL(for_variable);
  CHECK_BOOL(while_variable);
  CHECK_BOOL(do_condition);
  CHECK_BOOL(switch_condition);
  CHECK_VALUE(receiver);
  CHECK_VALUE(tagged);
  CHECK_VALUE(operator_call);
  CHECK_VALUE(rvalue_argument);
  CHECK_VALUE(cast_rvalue_argument);
  CHECK_VALUE(rvalue_alias);
  CHECK_VALUE(ordinary_condition);
  CHECK_VALUE(cast_operands);
  CHECK_BOOL(opaque_results);
  CHECK_VALUE(comma_operand);
}
// EXEC: argument_conditional 0 0.0 0
// EXEC-NEXT: argument_conditional 1 4.0 1
// EXEC-NEXT: argument_logical 0 0.0 0
// EXEC-NEXT: argument_logical 1 4.0 1
// EXEC-NEXT: nested_conditional 0 0.0 0
// EXEC-NEXT: nested_conditional 1 4.0 1
// EXEC-NEXT: lazy_product 0 0.0 0
// EXEC-NEXT: lazy_product 1 4.0 1
// EXEC-NEXT: selected_product 0 4.0 0
// EXEC-NEXT: selected_product 1 8.0 1
// EXEC-NEXT: if_condition 0 0.0 0
// EXEC-NEXT: if_condition 1 4.0 1
// EXEC-NEXT: loop_condition 0 0.0 0
// EXEC-NEXT: loop_condition 1 6.0 2
// EXEC-NEXT: for_variable 0 0.0 0
// EXEC-NEXT: for_variable 1 6.0 2
// EXEC-NEXT: while_variable 0 0.0 0
// EXEC-NEXT: while_variable 1 6.0 2
// EXEC-NEXT: do_condition 0 0.0 0
// EXEC-NEXT: do_condition 1 4.0 2
// EXEC-NEXT: switch_condition 0 0.0 0
// EXEC-NEXT: switch_condition 1 4.0 1
// EXEC-NEXT: receiver 4.0 0
// EXEC-NEXT: tagged 4.0 0
// EXEC-NEXT: operator_call 4.0 0
// EXEC-NEXT: rvalue_argument 4.0 1
// EXEC-NEXT: cast_rvalue_argument 4.0 1
// EXEC-NEXT: rvalue_alias 4.0 1
// EXEC-NEXT: ordinary_condition 1.0 0
// EXEC-NEXT: cast_operands 4.0 5
// EXEC-NEXT: opaque_results 0 1.0 0
// EXEC-NEXT: opaque_results 1 1.0 1
// EXEC-NEXT: comma_operand 4.0 2

// SCALAR-LABEL: inline clad::ValueAndPushforward<double, double> bump_pushforward(double x, double _d_x) {
// SCALAR-NEXT:     ++calls;
// SCALAR-NEXT:     return {x * x, _d_x * x + x * _d_x};
// SCALAR-NEXT: }
// SCALAR-LABEL: double argument_conditional_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     opaque(enabled ? [&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// SCALAR-NEXT:     }() : 0.);
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double argument_logical_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     opaque(enabled && [&]() -> bool {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// SCALAR-NEXT:     }());
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double nested_conditional_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     opaque(enabled ? enabled ? [&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// SCALAR-NEXT:     }() : 0. : 0.);
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double lazy_product_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     bool _d_unused = 0;
// SCALAR-NEXT:     bool unused = enabled && [&]() -> bool {
// SCALAR-NEXT:         double _t5 = opaque([&]() -> double {
// SCALAR-NEXT:             clad::ValueAndPushforward<double, double> _t4 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:             return (_d_r = _t4.pushforward) , (r = _t4.value);
// SCALAR-NEXT:         }());
// SCALAR-NEXT:         return (_t5 * x);
// SCALAR-NEXT:     }();
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double selected_product_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     clad::ValueAndPushforward<double, double> _cond0 = enabled ? [&]() -> clad::ValueAndPushforward<double, double> {
// SCALAR-NEXT:         double _t5 = opaque([&]() -> double {
// SCALAR-NEXT:             clad::ValueAndPushforward<double, double> _t4 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:             return (_d_r = _t4.pushforward) , (r = _t4.value);
// SCALAR-NEXT:         }());
// SCALAR-NEXT:         double _d_cond = _t5 * _d_x;
// SCALAR-NEXT:         return {_t5 * x, _d_cond};
// SCALAR-NEXT:     }() : [&]() -> clad::ValueAndPushforward<double, double> {
// SCALAR-NEXT:         double _d_cond = _d_x * x + x * _d_x;
// SCALAR-NEXT:         return {x * x, _d_cond};
// SCALAR-NEXT:     }();
// SCALAR-NEXT:     double _d_value = _cond0.pushforward;
// SCALAR-NEXT:     double value = _cond0.value;
// SCALAR-NEXT:     return _d_r + _d_value;
// SCALAR-NEXT: }
// SCALAR-LABEL: double if_condition_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     if (enabled && opaque([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// SCALAR-NEXT:     }())) {
// SCALAR-NEXT:     }
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double loop_condition_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     int _d_n = 0;
// SCALAR-NEXT:     int n = 0;
// SCALAR-NEXT:     while ((enabled && (n++ < 2)) && opaque([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// SCALAR-NEXT:     }()))
// SCALAR-NEXT:         {
// SCALAR-NEXT:             _d_x += 0;
// SCALAR-NEXT:             x += 1;
// SCALAR-NEXT:         }
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double for_variable_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     int _d_n = 0;
// SCALAR-NEXT:     int n = 0;
// SCALAR-NEXT:     {
// SCALAR-NEXT:         double _d_t;
// SCALAR-NEXT:         double t;
// SCALAR-NEXT:         for (; [&]() -> double & {
// SCALAR-NEXT:             bool _t5 = enabled && (n++ < 2);
// SCALAR-NEXT:             return (_d_t = _t5 ? 0 : 0.) , (t = _t5 ? opaque([&]() -> double {
// SCALAR-NEXT:                 clad::ValueAndPushforward<double, double> _t4 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:                 return (_d_r = _t4.pushforward) , (r = _t4.value);
// SCALAR-NEXT:             }()) : 0.);
// SCALAR-NEXT:         }();) {
// SCALAR-NEXT:             _d_x += 0;
// SCALAR-NEXT:             x += 1;
// SCALAR-NEXT:         }
// SCALAR-NEXT:     }
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double while_variable_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     int _d_n = 0;
// SCALAR-NEXT:     int n = 0;
// SCALAR-NEXT:     double _d_t;
// SCALAR-NEXT:     double t;
// SCALAR-NEXT:     while ([&]() -> double & {
// SCALAR-NEXT:         bool _t5 = enabled && (n++ < 2);
// SCALAR-NEXT:         return (_d_t = _t5 ? 0 : 0.) , (t = _t5 ? opaque([&]() -> double {
// SCALAR-NEXT:             clad::ValueAndPushforward<double, double> _t4 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:             return (_d_r = _t4.pushforward) , (r = _t4.value);
// SCALAR-NEXT:         }()) : 0.);
// SCALAR-NEXT:     }())
// SCALAR-NEXT:         {
// SCALAR-NEXT:             _d_x += 0;
// SCALAR-NEXT:             x += 1;
// SCALAR-NEXT:         }
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double do_condition_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     int _d_n = 0;
// SCALAR-NEXT:     int n = 0;
// SCALAR-NEXT:     do {
// SCALAR-NEXT:     } while ((enabled && (n++ < 2)) && opaque([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// SCALAR-NEXT:     }()));
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double switch_condition_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     {
// SCALAR-NEXT:         switch (enabled ? static_cast<int>(opaque([&]() -> double {
// SCALAR-NEXT:             clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:             return (_d_r = _t1.pushforward) , (r = _t1.value);
// SCALAR-NEXT:         }())) : 0) {
// SCALAR-NEXT:           default:
// SCALAR-NEXT:             {
// SCALAR-NEXT:                 break;
// SCALAR-NEXT:             }
// SCALAR-NEXT:         }
// SCALAR-NEXT:     }
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double receiver_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     Sink _d_s;
// SCALAR-NEXT:     Sink s;
// SCALAR-NEXT:     ((_d_r = _d_x * x + x * _d_x) , (r = x * x)) , s.ignore(0.);
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double tagged_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     TaggedSink _d_s;
// SCALAR-NEXT:     TaggedSink s;
// SCALAR-NEXT:     s.ignore((_d_r = _d_x * x + x * _d_x) , (r = x * x));
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double operator_call_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     Functor _d_s;
// SCALAR-NEXT:     Functor s;
// SCALAR-NEXT:     s((_d_r = _d_x * x + x * _d_x) , (r = x * x));
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double rvalue_argument_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     take_rvalue(static_cast<double &&>([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(r = x * x, _d_r = _d_x * x + x * _d_x);
// SCALAR-NEXT:         return static_cast<double &&>(_t1.value);
// SCALAR-NEXT:     }()));
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double cast_rvalue_argument_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     take_rvalue(static_cast<double &&>(static_cast<double &&>([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(r = x * x, _d_r = _d_x * x + x * _d_x);
// SCALAR-NEXT:         return static_cast<double &&>(_t1.value);
// SCALAR-NEXT:     }())));
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double rvalue_alias_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     take_rvalue(static_cast<double &&>([&]() -> double & {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// SCALAR-NEXT:     }()));
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double ordinary_condition_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_y = _d_x;
// SCALAR-NEXT:     double y = x;
// SCALAR-NEXT:     if (y += y) {
// SCALAR-NEXT:     }
// SCALAR-NEXT:     return _d_y;
// SCALAR-NEXT: }
// SCALAR-LABEL: double cast_operands_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     take_rvalue(static_cast<double &&>(static_cast<double &&>([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(r = x * x, _d_r = _d_x * x + x * _d_x);
// SCALAR-NEXT:         return static_cast<double &&>(_t1.value);
// SCALAR-NEXT:     }())));
// SCALAR-NEXT:     take_rvalue((double &&)static_cast<double &&>([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t3 = bump_pushforward(r = x * x, _d_r = _d_x * x + x * _d_x);
// SCALAR-NEXT:         return static_cast<double &&>(_t3.value);
// SCALAR-NEXT:     }()));
// SCALAR-NEXT:     opaque(double([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t5 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t5.pushforward) , (r = _t5.value);
// SCALAR-NEXT:     }()));
// SCALAR-NEXT:     take_rvalue(const_cast<double &&>([&]() -> double & {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t7 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t7.pushforward) , (r = _t7.value);
// SCALAR-NEXT:     }()));
// SCALAR-NEXT:     take_rvalue(reinterpret_cast<double &&>([&]() -> double & {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t9 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t9.pushforward) , (r = _t9.value);
// SCALAR-NEXT:     }()));
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double opaque_results_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_value = 0;
// SCALAR-NEXT:     double value = opaque(x);
// SCALAR-NEXT:     bool _d_flag = enabled ? 0 : 0;
// SCALAR-NEXT:     bool flag = enabled ? opaque_bool(x) : false;
// SCALAR-NEXT:     return _d_x + _d_value + (flag ? _d_value : 0.);
// SCALAR-NEXT: }
// SCALAR-LABEL: double comma_operand_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     opaque(counted([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// SCALAR-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// SCALAR-NEXT:     }()) , 0.);
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }

// VECTOR-LABEL: inline clad::ValueAndPushforward<double, clad::array<double> > bump_vector_pushforward(double x, clad::array<double> _d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = _d_x.size();
// VECTOR-NEXT:     ++calls;
// VECTOR-NEXT:     return {x * x, _d_x * x + x * _d_x};
// VECTOR-NEXT: }
// VECTOR-LABEL: void argument_conditional_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     opaque(enabled ? [&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t1.pushforward) , (r = _t1.value);
// VECTOR-NEXT:     }() : (clad::zero_vector(indepVarCount)) , 0.);
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void argument_logical_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     opaque(enabled && [&]() -> bool {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t1.pushforward) , (r = _t1.value);
// VECTOR-NEXT:     }());
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void nested_conditional_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     opaque(enabled ? enabled ? [&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t1.pushforward) , (r = _t1.value);
// VECTOR-NEXT:     }() : (clad::zero_vector(indepVarCount)) , 0. : (clad::zero_vector(indepVarCount)) , 0.);
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void lazy_product_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::array<bool> _d_vector_unused(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     bool unused = enabled && [&]() -> bool {
// VECTOR-NEXT:         double _t5 = opaque([&]() -> double {
// VECTOR-NEXT:             clad::ValueAndPushforward<double, clad::array<double> > _t4 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:             return (_d_vector_r = _t4.pushforward) , (r = _t4.value);
// VECTOR-NEXT:         }());
// VECTOR-NEXT:         return ((clad::zero_vector(indepVarCount)) * x + _t5 * _d_vector_x) , (_t5 * x);
// VECTOR-NEXT:     }();
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void selected_product_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::ValueAndPushforward<double, clad::array<double> > _cond0 = enabled ? [&]() -> clad::ValueAndPushforward<double, clad::array<double> > {
// VECTOR-NEXT:         double _t5 = opaque([&]() -> double {
// VECTOR-NEXT:             clad::ValueAndPushforward<double, clad::array<double> > _t4 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:             return (_d_vector_r = _t4.pushforward) , (r = _t4.value);
// VECTOR-NEXT:         }());
// VECTOR-NEXT:         clad::array<double> _d_cond = (clad::zero_vector(indepVarCount)) * x + _t5 * _d_vector_x;
// VECTOR-NEXT:         return {_t5 * x, _d_cond};
// VECTOR-NEXT:     }() : [&]() -> clad::ValueAndPushforward<double, clad::array<double> > {
// VECTOR-NEXT:         clad::array<double> _d_cond = _d_vector_x * x + x * _d_vector_x;
// VECTOR-NEXT:         return {x * x, _d_cond};
// VECTOR-NEXT:     }();
// VECTOR-NEXT:     clad::array<double> _d_vector_value(_cond0.pushforward);
// VECTOR-NEXT:     double value = _cond0.value;
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r + _d_vector_value);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void if_condition_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     if (enabled && ((clad::zero_vector(indepVarCount)) , opaque([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t1.pushforward) , (r = _t1.value);
// VECTOR-NEXT:     }()))) {
// VECTOR-NEXT:     }
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void loop_condition_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::array<int> _d_vector_n(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     int n = 0;
// VECTOR-NEXT:     while ((enabled && (n++ < ((clad::zero_vector(indepVarCount)) , 2))) && ((clad::zero_vector(indepVarCount)) , opaque([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t1.pushforward) , (r = _t1.value);
// VECTOR-NEXT:     }())))
// VECTOR-NEXT:         {
// VECTOR-NEXT:             _d_vector_x += clad::zero_vector(indepVarCount);
// VECTOR-NEXT:             x += 1;
// VECTOR-NEXT:         }
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void for_variable_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::array<int> _d_vector_n(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     int n = 0;
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_t = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:         double t;
// VECTOR-NEXT:         for (; [&]() -> double & {
// VECTOR-NEXT:             bool _t5 = enabled && (n++ < ((clad::zero_vector(indepVarCount)) , 2));
// VECTOR-NEXT:             return (_d_t = _t5 ? clad::zero_vector(indepVarCount) : clad::zero_vector(indepVarCount)) , (t = _t5 ? opaque([&]() -> double {
// VECTOR-NEXT:                 clad::ValueAndPushforward<double, clad::array<double> > _t4 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:                 return (_d_vector_r = _t4.pushforward) , (r = _t4.value);
// VECTOR-NEXT:             }()) : 0.);
// VECTOR-NEXT:         }();) {
// VECTOR-NEXT:             _d_vector_x += clad::zero_vector(indepVarCount);
// VECTOR-NEXT:             x += 1;
// VECTOR-NEXT:         }
// VECTOR-NEXT:     }
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void while_variable_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::array<int> _d_vector_n(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     int n = 0;
// VECTOR-NEXT:     clad::array<double> _d_t = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     double t;
// VECTOR-NEXT:     while ([&]() -> double & {
// VECTOR-NEXT:         bool _t5 = enabled && (n++ < ((clad::zero_vector(indepVarCount)) , 2));
// VECTOR-NEXT:         return (_d_t = _t5 ? clad::zero_vector(indepVarCount) : clad::zero_vector(indepVarCount)) , (t = _t5 ? opaque([&]() -> double {
// VECTOR-NEXT:             clad::ValueAndPushforward<double, clad::array<double> > _t4 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:             return (_d_vector_r = _t4.pushforward) , (r = _t4.value);
// VECTOR-NEXT:         }()) : 0.);
// VECTOR-NEXT:     }())
// VECTOR-NEXT:         {
// VECTOR-NEXT:             _d_vector_x += clad::zero_vector(indepVarCount);
// VECTOR-NEXT:             x += 1;
// VECTOR-NEXT:         }
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void do_condition_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::array<int> _d_vector_n(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     int n = 0;
// VECTOR-NEXT:     do {
// VECTOR-NEXT:     } while ((enabled && (n++ < ((clad::zero_vector(indepVarCount)) , 2))) && ((clad::zero_vector(indepVarCount)) , opaque([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t1.pushforward) , (r = _t1.value);
// VECTOR-NEXT:     }())));
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void switch_condition_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     {
// VECTOR-NEXT:         switch (enabled ? static_cast<int>((clad::zero_vector(indepVarCount)) , opaque([&]() -> double {
// VECTOR-NEXT:             clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:             return (_d_vector_r = _t1.pushforward) , (r = _t1.value);
// VECTOR-NEXT:         }())) : (clad::zero_vector(indepVarCount)) , 0) {
// VECTOR-NEXT:           default:
// VECTOR-NEXT:             {
// VECTOR-NEXT:                 break;
// VECTOR-NEXT:             }
// VECTOR-NEXT:         }
// VECTOR-NEXT:     }
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void receiver_dvec(double x, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::array<Sink> _d_vector_s;
// VECTOR-NEXT:     Sink s;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     ((_d_vector_r = _d_vector_x * x + x * _d_vector_x) , (r = x * x)) , s.ignore((clad::zero_vector(indepVarCount)) , 0.);
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void tagged_dvec(double x, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::array<TaggedSink> _d_vector_s;
// VECTOR-NEXT:     TaggedSink s;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     s.ignore((_d_vector_r = _d_vector_x * x + x * _d_vector_x) , (r = x * x));
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void operator_call_dvec(double x, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::array<Functor> _d_vector_s;
// VECTOR-NEXT:     Functor s;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     s((_d_vector_r = _d_vector_x * x + x * _d_vector_x) , (r = x * x));
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void rvalue_argument_dvec(double x, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     take_rvalue(static_cast<double &&>([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(r = x * x, _d_vector_r = _d_vector_x * x + x * _d_vector_x);
// VECTOR-NEXT:         return static_cast<double &&>(_t1.value);
// VECTOR-NEXT:     }()));
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void cast_rvalue_argument_dvec(double x, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     take_rvalue(static_cast<double &&>(static_cast<double &&>([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(r = x * x, _d_vector_r = _d_vector_x * x + x * _d_vector_x);
// VECTOR-NEXT:         return static_cast<double &&>(_t1.value);
// VECTOR-NEXT:     }())));
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void rvalue_alias_dvec(double x, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     take_rvalue(static_cast<double &&>([&]() -> double & {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t1.pushforward) , (r = _t1.value);
// VECTOR-NEXT:     }()));
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void ordinary_condition_dvec(double x, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<double> _d_vector_y(_d_vector_x);
// VECTOR-NEXT:     double y = x;
// VECTOR-NEXT:     if (y += y) {
// VECTOR-NEXT:     }
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_y);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void cast_operands_dvec(double x, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     take_rvalue(static_cast<double &&>(static_cast<double &&>([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(r = x * x, _d_vector_r = _d_vector_x * x + x * _d_vector_x);
// VECTOR-NEXT:         return static_cast<double &&>(_t1.value);
// VECTOR-NEXT:     }())));
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     take_rvalue((double &&)static_cast<double &&>([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t3 = bump_vector_pushforward(r = x * x, _d_vector_r = _d_vector_x * x + x * _d_vector_x);
// VECTOR-NEXT:         return static_cast<double &&>(_t3.value);
// VECTOR-NEXT:     }()));
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     opaque(double([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t5 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t5.pushforward) , (r = _t5.value);
// VECTOR-NEXT:     }()));
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     take_rvalue(const_cast<double &&>([&]() -> double & {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t7 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t7.pushforward) , (r = _t7.value);
// VECTOR-NEXT:     }()));
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     take_rvalue(reinterpret_cast<double &&>([&]() -> double & {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t9 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t9.pushforward) , (r = _t9.value);
// VECTOR-NEXT:     }()));
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void opaque_results_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_value(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double value = opaque(x);
// VECTOR-NEXT:     clad::array<bool> _d_vector_flag(enabled ? clad::zero_vector(indepVarCount) : clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     bool flag = enabled ? opaque_bool(x) : false;
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_x + _d_vector_value + (flag ? _d_vector_value : clad::zero_vector(indepVarCount)));
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void comma_operand_dvec(double x, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     opaque(((clad::zero_vector(indepVarCount)) , counted([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = bump_vector_pushforward(x, _d_vector_x);
// VECTOR-NEXT:         return (_d_vector_r = _t1.pushforward) , (r = _t1.value);
// VECTOR-NEXT:     }())) , ((clad::zero_vector(indepVarCount)) , 0.));
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
