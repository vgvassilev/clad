// RUN: %cladclang %s -I%S/../../include -o %t -Xclang -verify 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

CLAD_NONDIFFERENTIABLE void ignore(double, double) {}
CLAD_NONDIFFERENTIABLE double opaque(double value) { return value * 7; }
struct Sink {
  CLAD_NONDIFFERENTIABLE void ignore(double) const {}
};

double direct(double x, double y) {
  double r = 0, s = 0;
  ignore(r = x * y, s = x + y);
  return r + s;
}

double nested(double x) {
  double r = 0;
  double value = 2 * opaque(r = x * x);
  return r + value;
}

double looping(double x) {
  double r = 0;
  for (int i = 0; i < 3; ++i) {
    ignore(r += x * x, 0.);
    x += 1;
  }
  return r;
}

double returned(double x) {
  return (opaque(x = x * x), x);
}

double discrete(double x) {
  int n = 2;
  return (opaque(n++), x * n);
}

double shorted(double x, bool enabled) {
  double r = 0;
  bool unused = enabled && opaque(r = x * x);
  return r;
}

double conditional(double x, bool enabled) {
  double r = 0;
  double unused = enabled ? opaque(r = x * x) : 0.;
  return r;
}

double member(double x) {
  Sink s;
  double r = 0;
  s.ignore(r = x * x);
  return r;
}

int ref_calls = 0;
CLAD_NONDIFFERENTIABLE double& opaque_ref(double& value) {
  ++ref_calls;
  return value;
}

struct Storage {
  double values[2];
  unsigned flag : 3;
};

double shapes(double x) {
  Storage data = {};
  double* p = data.values;
  ignore(data.flag = 2, 0.);
  opaque_ref(*p++ = x * x);
  opaque_ref(*p = x);
  return data.values[0] + data.values[1];
}

int bump_calls = 0;
double bump(double v) {
  ++bump_calls;
  return v * v;
}

CLAD_NONDIFFERENTIABLE double&& opaque_move(double& value) {
  ++ref_calls;
  return static_cast<double&&>(value);
}

double nested_call(double x, bool enabled) {
  double r = 0;
  bool unused = enabled && opaque(r = bump(x));
  return r;
}

double nested_void(double x, bool enabled) {
  double r = 0;
  bool unused = enabled && (ignore(r = bump(x), 0.), true);
  return r;
}

double nested_reference(double x, bool enabled) {
  double r = 0;
  bool unused = enabled && opaque_ref(r = bump(x));
  return r;
}

double nested_rvalue(double x, bool enabled) {
  double r = 0;
  bool unused = enabled && opaque_move(r = bump(x));
  return r;
}

double discarded_reference(double x, bool enabled) {
  double r = 0;
  if (enabled)
    opaque_ref(r = bump(x));
  return r;
}

double nested_conditional(double x, bool enabled) {
  double r = 0;
  double unused = enabled ? opaque(r = bump(x)) : 0.;
  return r;
}

double nested_condition(double x, int n) {
  double r = 0;
  for (int i = 0; i < n && opaque(r = bump(x)); ++i)
    x += 1;
  return r;
}

int address_calls = 0;
struct Box {
  double value = 1;
  Box* operator&() {
    ++address_calls;
    return nullptr;
  }
};
CLAD_NONDIFFERENTIABLE Box& opaque_box(Box& box, double) { return box; }

double object_reference(double x, bool enabled) {
  double r = 0;
  Box box;
  if (enabled)
    opaque_box(box, r = bump(x));
  return r;
}

struct Acc {
  double v;
  Acc& operator=(const Acc& other) {
    v = other.v;
    return *this;
  }
  Acc& operator++() {
    v += v;
    return *this;
  }
};
CLAD_NONDIFFERENTIABLE void opaque_acc(const Acc&) {}

double overloaded_assign(double x) {
  Acc a{x}, b{x * x};
  opaque_acc(a = b);
  return a.v;
}

double overloaded_inc(double x) {
  Acc a{x};
  opaque_acc(++a);
  return a.v;
}

int main() {
  auto d = clad::differentiate(direct, "x");
  auto n = clad::differentiate(nested, "x");
  auto l = clad::differentiate(looping, "x");
  auto m = clad::differentiate(member, "x");
  auto r = clad::differentiate(returned, "x");
  auto t = clad::differentiate(shapes, "x");
  auto i = clad::differentiate(discrete, "x");
  auto a = clad::differentiate(shorted, "x");
  auto b = clad::differentiate(conditional, "x");
  auto c = clad::differentiate(nested_call, "x");
  auto v = clad::differentiate(nested_void, "x");
  auto q = clad::differentiate(nested_reference, "x");
  auto z = clad::differentiate(nested_rvalue, "x");
  auto f = clad::differentiate(discarded_reference, "x");
  auto e = clad::differentiate(nested_conditional, "x");
  auto k = clad::differentiate(nested_condition, "x");
  auto o = clad::differentiate(object_reference, "x");
  double dx = 0, dy = 0;
  clad::gradient(direct).execute(2., 3., &dx, &dy);
  printf("direct %.1f %.1f %.1f\n", d.execute(2., 3.), dx, dy);
  printf("nested %.1f\n", n.execute(2.));
  printf("loop %.1f\n", l.execute(2.));
  printf("member %.1f\n", m.execute(2.));
  printf("returned %.1f\n", r.execute(2.));
  ref_calls = 0;
  double result = t.execute(2.);
  printf("shapes %.1f %d\n", result, ref_calls);
  printf("discrete %.1f\n", i.execute(2.));
  printf("short %.1f %.1f\n", a.execute(2., false), a.execute(2., true));
  printf("conditional %.1f %.1f\n", b.execute(2., false), b.execute(2., true));
  bump_calls = 0;
  double off = c.execute(2., false);
  int off_calls = bump_calls;
  bump_calls = 0;
  double on = c.execute(2., true);
  printf("nested call %.1f %d %.1f %d\n", off, off_calls, on, bump_calls);
  printf("nested void %.1f %.1f\n", v.execute(2., false), v.execute(2., true));
  printf("nested reference %.1f %.1f\n", q.execute(2., false), q.execute(2., true));
  printf("nested rvalue %.1f %.1f\n", z.execute(2., false), z.execute(2., true));
  printf("discarded reference %.1f %.1f\n", f.execute(2., false), f.execute(2., true));
  printf("nested conditional %.1f %.1f\n", e.execute(2., false), e.execute(2., true));
  bump_calls = 0;
  off = k.execute(2., 0);
  off_calls = bump_calls;
  bump_calls = 0;
  on = k.execute(2., 3);
  printf("nested condition %.1f %d %.1f %d\n", off, off_calls, on, bump_calls);
  address_calls = 0;
  off = o.execute(2., false);
  on = o.execute(2., true);
  printf("object reference %.1f %.1f %d\n", off, on, address_calls);
  auto assign = clad::differentiate(overloaded_assign, "x");
  auto increment = clad::differentiate(overloaded_inc, "x");
  printf("operators %.1f %.1f\n", assign.execute(2.), increment.execute(2.));
  // CHECK-EXEC: direct 4.0 4.0 3.0
  // CHECK-EXEC-NEXT: nested 4.0
  // CHECK-EXEC-NEXT: loop 18.0
  // CHECK-EXEC-NEXT: member 4.0
  // CHECK-EXEC-NEXT: returned 4.0
  // CHECK-EXEC-NEXT: shapes 5.0 2
  // CHECK-EXEC-NEXT: discrete 3.0
  // CHECK-EXEC-NEXT: short 0.0 4.0
  // CHECK-EXEC-NEXT: conditional 0.0 4.0
  // CHECK-EXEC-NEXT: nested call 0.0 0 4.0 1
  // CHECK-EXEC-NEXT: nested void 0.0 4.0
  // CHECK-EXEC-NEXT: nested reference 0.0 4.0
  // CHECK-EXEC-NEXT: nested rvalue 0.0 4.0
  // CHECK-EXEC-NEXT: discarded reference 0.0 4.0
  // CHECK-EXEC-NEXT: nested conditional 0.0 4.0
  // CHECK-EXEC-NEXT: nested condition 0.0 0 8.0 3
  // CHECK-EXEC-NEXT: object reference 0.0 4.0 0
  // CHECK-EXEC-NEXT: operators 4.0 2.0
}

// Generated derivative bodies.
// CHECK-LABEL: double direct_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     double _d_r = 0, _d_s = 0;
// CHECK-NEXT:     double r = 0, s = 0;
// CHECK-NEXT:     ignore((_d_r = _d_x * y + x * _d_y) , (r = x * y), (_d_s = _d_x + _d_y) , (s = x + y));
// CHECK-NEXT:     return _d_r + _d_s;
// CHECK-NEXT: }
// CHECK-LABEL: double nested_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     double _d_value = 0.;
// CHECK-NEXT:     double value = 2 * opaque((_d_r = _d_x * x + x * _d_x) , (r = x * x));
// CHECK-NEXT:     return _d_r + _d_value;
// CHECK-NEXT: }
// CHECK-LABEL: double looping_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_i = 0;
// CHECK-NEXT:         for (int i = 0; i < 3; ++i) {
// CHECK-NEXT:             ignore((_d_r += _d_x * x + x * _d_x) , (r += x * x), 0.);
// CHECK-NEXT:             _d_x += 0;
// CHECK-NEXT:             x += 1;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: double member_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Sink _d_s;
// CHECK-NEXT:     Sink s;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     s.ignore((_d_r = _d_x * x + x * _d_x) , (r = x * x));
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: double returned_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     return (opaque((_d_x = _d_x * x + x * _d_x) , (x = x * x)) , _d_x);
// CHECK-NEXT: }
// CHECK-LABEL: double shapes_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Storage _d_data = {/*implicit*/(double[2])0, /*implicit*/(unsigned int)0};
// CHECK-NEXT:     Storage data = {/*implicit*/(double[2])0, /*implicit*/(unsigned int)0};
// CHECK-NEXT:     double *_d_p = _d_data.values;
// CHECK-NEXT:     double *p = data.values;
// CHECK-NEXT:     ignore((_d_data.flag = 0) , (data.flag = 2), 0.);
// CHECK-NEXT:     opaque_ref((*_d_p++ = _d_x * x + x * _d_x) , (*p++ = x * x));
// CHECK-NEXT:     opaque_ref((*_d_p = _d_x) , (*p = x));
// CHECK-NEXT:     return _d_data.values[0] + _d_data.values[1];
// CHECK-NEXT: }
// CHECK-LABEL: double discrete_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 2;
// CHECK-NEXT:     return (opaque(n++) , (_d_x * n + x * _d_n));
// CHECK-NEXT: }
// CHECK-LABEL: double shorted_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     bool _d_unused = 0;
// CHECK-NEXT:     bool unused = enabled && opaque((_d_r = _d_x * x + x * _d_x) , (r = x * x));
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: double conditional_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     double _d_unused = enabled ? 0 : 0.;
// CHECK-NEXT:     double unused = enabled ? opaque((_d_r = _d_x * x + x * _d_x) , (r = x * x)) : 0.;
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: inline clad::ValueAndPushforward<double, double> bump_pushforward(double v, double _d_v) {
// CHECK-NEXT:     ++bump_calls;
// CHECK-NEXT:     return {v * v, _d_v * v + v * _d_v};
// CHECK-NEXT: }
// CHECK-LABEL: double nested_call_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     bool _d_unused = 0;
// CHECK-NEXT:     bool unused = enabled && opaque([&]() -> double {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// CHECK-NEXT:     }());
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: double nested_void_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     bool _d_unused = 0;
// CHECK-NEXT:     bool unused = enabled && (ignore([&]() -> double {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// CHECK-NEXT:     }(), 0.) , true);
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: double nested_reference_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     bool _d_unused = 0;
// CHECK-NEXT:     bool unused = enabled && opaque_ref([&]() -> double & {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// CHECK-NEXT:     }());
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: double nested_rvalue_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     bool _d_unused = 0;
// CHECK-NEXT:     bool unused = enabled && opaque_move([&]() -> double & {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// CHECK-NEXT:     }());
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: double discarded_reference_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     if (enabled)
// CHECK-NEXT:         opaque_ref([&]() -> double & {
// CHECK-NEXT:             clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:             return (_d_r = _t1.pushforward) , (r = _t1.value);
// CHECK-NEXT:         }());
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: double nested_conditional_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     double _d_unused = enabled ? 0 : 0.;
// CHECK-NEXT:     double unused = enabled ? opaque([&]() -> double {
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:         return (_d_r = _t1.pushforward) , (r = _t1.value);
// CHECK-NEXT:     }()) : 0.;
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: double nested_condition_darg0(double x, int n) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_i = 0;
// CHECK-NEXT:         for (int i = 0; (i < n) && opaque([&]() -> double {
// CHECK-NEXT:             clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:             return (_d_r = _t1.pushforward) , (r = _t1.value);
// CHECK-NEXT:         }()); ++i) {
// CHECK-NEXT:             _d_x += 0;
// CHECK-NEXT:             x += 1;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: double object_reference_darg0(double x, bool enabled) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_enabled = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     Box _d_box;
// CHECK-NEXT:     Box box;
// CHECK-NEXT:     if (enabled)
// CHECK-NEXT:         opaque_box(box, [&]() -> double {
// CHECK-NEXT:             clad::ValueAndPushforward<double, double> _t1 = bump_pushforward(x, _d_x);
// CHECK-NEXT:             return (_d_r = _t1.pushforward) , (r = _t1.value);
// CHECK-NEXT:         }());
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
// CHECK-LABEL: void direct_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     double _d_r = 0., _d_s = 0.;
// CHECK-NEXT:     double r = 0, s = 0;
// CHECK-NEXT:     ignore(r = x * y, s = x + y);
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_r += 1;
// CHECK-NEXT:         _d_s += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += _d_r * y;
// CHECK-NEXT:         *_d_y += x * _d_r;
// CHECK-NEXT:         _d_r = 0.;
// CHECK-NEXT:         *_d_x += _d_s;
// CHECK-NEXT:         *_d_y += _d_s;
// CHECK-NEXT:         _d_s = 0.;
// CHECK-NEXT:     }
// CHECK-NEXT: }
// CHECK-LABEL: inline clad::ValueAndPushforward<Acc &, Acc &> operator_equal_pushforward(const Acc &other, Acc *_d_this, const Acc &_d_other) {
// CHECK-NEXT:     _d_this->v = _d_other.v;
// CHECK-NEXT:     this->v = other.v;
// CHECK-NEXT:     return {(Acc &)*this, (Acc &)*_d_this};
// CHECK-NEXT: }
// CHECK-LABEL: double overloaded_assign_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Acc _d_a{_d_x}, _d_b{_d_x * x + x * _d_x};
// CHECK-NEXT:     Acc a{x}, b{x * x};
// CHECK-NEXT:     opaque_acc([&]() -> const Acc & {
// CHECK-NEXT:         clad::ValueAndPushforward<Acc &, Acc &> _t1 = a.operator_equal_pushforward(b, &_d_a, _d_b);
// CHECK-NEXT:         return _t1.value;
// CHECK-NEXT:     }());
// CHECK-NEXT:     return _d_a.v;
// CHECK-NEXT: }
// CHECK-LABEL: inline clad::ValueAndPushforward<Acc &, Acc &> operator_plus_plus_pushforward(Acc *_d_this) {
// CHECK-NEXT:     _d_this->v += _d_this->v;
// CHECK-NEXT:     this->v += this->v;
// CHECK-NEXT:     return {(Acc &)*this, (Acc &)*_d_this};
// CHECK-NEXT: }
// CHECK-LABEL: double overloaded_inc_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Acc _d_a{_d_x};
// CHECK-NEXT:     Acc a{x};
// CHECK-NEXT:     opaque_acc([&]() -> const Acc & {
// CHECK-NEXT:         clad::ValueAndPushforward<Acc &, Acc &> _t1 = a.operator_plus_plus_pushforward(&_d_a);
// CHECK-NEXT:         return _t1.value;
// CHECK-NEXT:     }());
// CHECK-NEXT:     return _d_a.v;
// CHECK-NEXT: }
