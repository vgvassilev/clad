// RUN: %cladclang %s -I%S/../../include -oActivityForward.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./ActivityForward.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oActivityForward.out -Xclang -verify
// RUN: ./ActivityForward.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-va %s -I%S/../../include -oActivityForward.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./ActivityForward.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

double h(double u) { return u * u; }

// The planner runs varied analysis over the direction the request seeds, so a
// call none of whose arguments can carry that direction gets no pushforward.
// Neither parameter is const: what decides is activity, not whether the
// tangent clad built happens to be a constant.
double two_calls(double x, double y) { return h(x) + h(y); }

// CHECK: double two_calls_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = h_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward + 0.;
// CHECK-NEXT: }

// A local that the direction reaches only through an assignment is varied all
// the same, so the call it feeds keeps its pushforward.
double through_local(double x, double y) {
  double t = y;
  t = t + x;
  return h(t);
}

// CHECK: double through_local_darg1(double x, double y) {
// CHECK-NEXT:     double _d_x = 0;
// CHECK-NEXT:     double _d_y = 1;
// CHECK-NEXT:     double _d_t = _d_y;
// CHECK-NEXT:     double t = y;
// CHECK-NEXT:     _d_t = _d_t + _d_x;
// CHECK-NEXT:     t = t + x;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = h_pushforward(t, _d_t);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// The analysis passes over a loop body more than once, and the first pass
// reaches the call before `t = x` makes t varied. The verdict has to be the
// one of the last pass, or the derivative loses the term.
double varied_late_in_loop(double x, double y) {
  double s = 0;
  double t = y;
  for (int i = 0; i < 2; ++i) {
    s += h(t);
    t = x;
  }
  return s;
}

// Object state is not tracked per field, so a member carrying the direction
// counts as varied and the call reading it keeps its pushforward.
struct MemberState {
  double c;
  double m(double x) {
    c = x;
    return h(c);
  }
};

// CHECK: double m_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     MemberState _d_this_obj;
// CHECK-NEXT:     MemberState *_d_this = &_d_this_obj;
// CHECK-NEXT:     _d_this->c = _d_x;
// CHECK-NEXT:     this->c = x;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = h_pushforward(this->c, _d_this->c);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// A functor differentiated with respect to one of its fields seeds no
// parameter at all; what it reads through `this` is varied all the same.
struct Functor {
  double x;
  double operator()() { return h(x); }
};

// CHECK: double operator_call_darg0() {
// CHECK-NEXT:     Functor _d_this_obj;
// CHECK-NEXT:     Functor *_d_this = &_d_this_obj;
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = h_pushforward(this->x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// A call through a function pointer has no callee declaration for the
// analysis to inspect; it must survive the analysis and stay in the primal.
int cb(int v) { return v; }
double with_indirect_call(double x, int (*f)(int)) {
  f(1); // expected-warning {{differentiation of indirect calls is not supported}}
  return x * x;
}

// CHECK: double with_indirect_call_darg0(double x, int (*f)(int)) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     f(1);
// CHECK-NEXT:     return _d_x * x + x * _d_x;
// CHECK-NEXT: }

// Activity is tracked per variable: seeding p[1] leaves the whole of p
// varied, and only folding the tangents built with the seeded index in hand
// recognizes h(p[0]) as inactive. See BaseForwardModeVisitor::VisitCallExpr.
double element_calls(double p[2]) { return h(p[0]) + h(p[1]); }

// CHECK: double element_calls_darg0_1(double p[2]) {
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = h_pushforward(p[1], 1.);
// CHECK-NEXT:     return 0. + _t0.pushforward;
// CHECK-NEXT: }

// An object built from the direction carries it through reference members.
// The analysis does not model constructor bodies, so a reference field's
// referent is unknown; the field is tracked as the object's own state,
// marked when a constructor argument is varied. The call handed the object
// keeps its pushforward. (The shape of a Kokkos functor capturing the
// independent variable by reference.)
struct RefCapture {
  double& out;
  double& x;
  RefCapture(double& _out, double& _x) : out(_out), x(_x) {}
  void operator()(int i) const { out = x * i; }
};

template <typename F> void run(int n, const F& f) { f(n); }

double through_ref_capture(double x) {
  double out = 0;
  RefCapture f(out, x);
  f(0); // puts RefCapture::operator() into the differentiation plan
  run(3, f);
  return out;
}

// CHECK: double through_ref_capture_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_out = 0;
// CHECK-NEXT:     double out = 0;
// CHECK-NEXT:     RefCapture _d_f{_d_out, _d_x};
// CHECK-NEXT:     RefCapture f{out, x};
// CHECK-NEXT:     f.operator_call_pushforward(0, &_d_f, 0);
// CHECK-NEXT:     run_pushforward(3, f, 0, _d_f);
// CHECK-NEXT:     return _d_out;
// CHECK-NEXT: }

int main() {
  auto d0 = clad::differentiate(two_calls, "x");
  printf("%.2f\n", d0.execute(3, 4)); // CHECK-EXEC: 6.00

  auto d1 = clad::differentiate(through_local, "y");
  printf("%.2f\n", d1.execute(3, 4)); // CHECK-EXEC: 14.00

  auto d2 = clad::differentiate(varied_late_in_loop, "x");
  printf("%.2f\n", d2.execute(3, 4)); // CHECK-EXEC: 6.00

  MemberState obj = {};
  auto d3 = clad::differentiate(&MemberState::m, "x");
  printf("%.2f\n", d3.execute(obj, 3)); // CHECK-EXEC: 6.00

  Functor fn = {3};
  auto d4 = clad::differentiate(fn, "x");
  printf("%.2f\n", d4.execute(fn)); // CHECK-EXEC: 6.00

  auto d5 = clad::differentiate(with_indirect_call, "x");
  printf("%.2f\n", d5.execute(3, cb)); // CHECK-EXEC: 6.00

  double p[2] = {3, 4};
  auto d6 = clad::differentiate(element_calls, "p[1]");
  printf("%.2f\n", d6.execute(p)); // CHECK-EXEC: 8.00

  auto d7 = clad::differentiate(through_ref_capture, "x");
  printf("%.2f\n", d7.execute(3)); // CHECK-EXEC: 3.00
}
