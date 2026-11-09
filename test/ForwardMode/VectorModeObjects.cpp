// RUN: %cladclang -fsyntax-only -ferror-limit=0 -std=c++17 \
// RUN:   -Xclang -verify=expected,refused -I%S/../../include %s
// RUN: %cladclang -DACCEPTED -std=c++17 -I%S/../../include %s \
// RUN:   -oVectorModeObjects.out 2>&1 | %filecheck %s
// RUN: ./VectorModeObjects.out | %filecheck_exec %s

// Vector mode gives an object one tangent per independent variable, each an
// object of its type: a clad::array of them. A method takes those of its
// object, and reads a member out of each. The second run builds and executes
// only what is accepted.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>
#include <utility>

struct M {
  double k;
  static double s;
  double sq(double x) const { return k * x * x; }
  double twice(double x) { return 2 * sq(x); }
  double getk() const { return k; }
};

// The reproducer from #2325. f = 2x^2 y.
double issue(double x, double y) {
  M m{2};
  return m.sq(x) * y;
}

// CHECK: clad::ValueAndPushforward<double, clad::array<double> > sq_vector_pushforward(double x, const clad::array<M> &_d_this, clad::array<double> _d_x) const {
// CHECK:     return {_t1 * x, ((clad::member_vector(_d_this, &M::k)) * x + _t0 * _d_x) * x + _t1 * _d_x};
// CHECK: void issue_dvec(double x, double y, double *_d_x, double *_d_y) {
// CHECK:     clad::array<M> _d_vector_m(clad::aggregate_vector(indepVarCount, (clad::array<double>)(clad::zero_vector(indepVarCount))));
// CHECK:     M m{2};
// CHECK:     clad::ValueAndPushforward<double, clad::array<double> > _t0 = m.sq_vector_pushforward(x, _d_vector_m, _d_vector_x);

// An object that depends on the independent variables. f = x y^2.
double dependent(double x, double y) {
  M m{x};
  return m.sq(y);
}

// A method calling another through `this`. f = 2 y x^2.
double through_this(double x, double y) {
  M m{y};
  return m.twice(x);
}

// A member read where the object is, and through a copy. f = x y.
double member_read(double x, double y) {
  M m{x};
  return m.k * y;
}
double copied(double x, double y) {
  M a{x};
  M b = a;
  return b.getk() * y;
}

// Every way to copy an object copies its tangents. f = xy.
double copies(double x, double y) {
  M a{x}, c{y};
  M b1 = M(a);
  M b2(a);
  M b3{a};
  M b4 = static_cast<M>(a);
  M b5 = x > 0 ? a : c;
  M b6 = std::move(a);
  return (b1.k + b2.k + b3.k + b4.k + b5.k + b6.k) / 6 * y;
}

// Assigning an object assigns its tangents. f = xy.
double assigned_object(double x, double y) {
  M a{1}, b{x};
  a = M{y};
  a = b;
  M c{1};
  c = a = static_cast<M&&>(b);
  return c.k * y;
}

// An object without an initializer has zero tangents. f = xy.
double uninitialized(double x, double y) {
  M m;
  (void)m;
  return x * y;
}

// Objects within objects, and members that are not numbers it differentiates.
struct In { double a; };
struct Out { In in; double b; int n; };
double nested(double x, double y) {
  Out o{{x}, y, 1};
  return o.in.a * o.b * o.n;
}

// A functor. f = 3xy, then f = xy.
struct F {
  double k;
  double operator()(double x) const { return k * x; }
};
double functor(double x, double y) {
  F fn{3};
  return fn(x) * y;
}
double functor_of_x(double x, double y) {
  F fn{x};
  return fn(y);
}

// A member left out is zero, and so is a static one. f = xy, then f = 5xy.
struct Two { In p; In q; };
double partial(double x, double y) {
  Two t{{x}};
  return (t.p.a + t.q.a) * y;
}
double M::s = 5;
double static_member(double x, double y) {
  M m{x};
  return m.s * m.k * y;
}

// An object without tangents, a global or what a global points to, has zero
// ones. f = 3x^2 y.
M gm{3};
M* gpm = &gm;
double global(double x, double y) { return gm.sq(x) * y; }
double global_pointer(double x, double y) { return gpm->sq(x) * y; }

// An object returned by a call has the tangents the call's pushforward
// returns: #2323. f = x^2 y, f = xy, f = xy^2.
struct V { double v; };
V mk(double x) { return V{x * x}; }
double issue_2323(double x, double y) { return mk(x).v * y; }
M make(double v) { return M{v}; }
// One made without the independent variables has zero ones. f = 4xy, then
// f = 4x^2 y.
M seed() { return M{4}; }
double seeded(double x, double y) {
  M m = seed();
  return m.k * x * y;
}
double seeded_call(double x, double y) { return seed().sq(x) * y; }

// A branch of ?: without tangents, beside one with them or not, has zero
// ones. f = xy, then f = 3xy.
double mixed_branches(double x, double y) {
  M a{x};
  M b = x > 0 ? a : seed();
  M c = x < 0 ? seed() : b;
  return c.k * y;
}
double no_branch(double x, double y) {
  M b = x > 0 ? gm : seed();
  return b.k * x * y;
}

// An object a lambda makes, or the last of a comma. f = xy.
double from_lambda(double x, double y) {
  auto make_m = [](double v) { return M{v}; };
  M a = make_m(x);
  M b = ((void)y, a);
  return b.k * y;
}
double returned(double x, double y) {
  M m = make(x);
  return m.k * y;
}
double returned_call(double x, double y) { return make(x).sq(y); }

// An object passed by value, made in the call. f = xy.
double use(M m, double x) { return m.k * x; }
double by_value(double x, double y) { return use(M{x}, y); }

// An object left at zero. f = (x + y) x.
double zero(double x, double y) {
  M m{};
  M n = M();
  return (m.k + n.k + x + y) * x;
}

#ifndef ACCEPTED
//===--- Refused ----------------------------------------------------------===//

// What is written to a member's tangents would be lost: they are read out of
// the object's, a copy.
double assigned(double x, double y) {
  M m{y};
  m.k = x; // refused-error {{vector mode cannot yet differentiate an assignment to a member of an object}}
  return m.k;
}
double added(double x, double y) {
  M m{y};
  m.k += x; // refused-error {{vector mode cannot yet differentiate an assignment to a member of an object}}
  return m.k;
}
double referenced(double x, double y) {
  M m{y};
  double& r = m.k; // refused-error {{vector mode cannot yet differentiate a reference to a member of an object}}
  r = x;
  return m.k;
}
struct S {
  double k;
  void set(double v) { k = v; } // refused-error {{vector mode cannot yet differentiate an assignment to a member of an object}}
  double bump() { return k + 1; }
};
double setter(double x, double y) {
  S s{y};
  s.set(x);
  return s.k;
}
struct HasS { S s; };
double member_method(double x, double y) {
  HasS h{{x}};
  return h.s.bump() * y; // refused-error {{vector mode cannot yet differentiate a call to a non-const method of a member of an object}}
}

// So would what is assigned to a member object, or to an object without
// tangents.
double assigned_member_object(double x, double y) {
  HasS h{{1}};
  h.s = S{x}; // refused-error {{vector mode cannot yet differentiate an assignment to a member of an object}}
  return h.s.k * y;
}
M gassigned{1};
double assigned_global(double x, double y) {
  gassigned = M{x}; // refused-error {{vector mode cannot yet differentiate an assignment to an object without a derivative}}
  return gassigned.k * y;
}

// An object reached through a pointer, and one without a default
// constructor.
double through_pointer(double x, M* p) { // refused-error {{vector mode cannot yet differentiate with respect to an object of type 'M'}}
  return p->k * x; // refused-warning {{'k' is accessed through an expression without a derivative; its derivative is treated as 0}}
}
double pointer(double x, double y) {
  M m{x};
  M* q = &m; // refused-error {{vector mode cannot yet differentiate a pointer to an object}}
  return q->k * y;
}
struct C {
  double k;
  C(double v) : k(v) {}
  double get() const { return k; }
};
double constructed(double x, double y) {
  C c(x); // refused-error {{vector mode cannot yet differentiate an object of type 'C', which has no default constructor}}
  return c.get() * y;
}
double object_array(double x, double y) {
  M arr[2] = {{x}, {y}}; // refused-error {{vector mode cannot yet differentiate an array of objects}}
  return arr[0].sq(x) * y; // refused-error {{vector mode cannot yet differentiate a call to a method of an object reached through a pointer or an array}}
}

void refused() {
  clad::differentiate<clad::opts::vector_mode>(assigned);
  clad::differentiate<clad::opts::vector_mode>(added);
  clad::differentiate<clad::opts::vector_mode>(referenced);
  clad::differentiate<clad::opts::vector_mode>(setter);
  clad::differentiate<clad::opts::vector_mode>(member_method);
  clad::differentiate<clad::opts::vector_mode>(assigned_member_object);
  clad::differentiate<clad::opts::vector_mode>(assigned_global);
  clad::differentiate<clad::opts::vector_mode>(through_pointer);
  clad::differentiate<clad::opts::vector_mode>(pointer);
  clad::differentiate<clad::opts::vector_mode>(object_array);
  clad::differentiate<clad::opts::vector_mode>(constructed);
}
#endif // ACCEPTED

#define TEST(F, dx, dy)                                                        \
  {                                                                            \
    auto d = clad::differentiate<clad::opts::vector_mode>(F);                  \
    double rx = 0, ry = 0;                                                     \
    d.execute(x, y, &rx, &ry);                                                 \
    printf("%s {%.2f, %.2f} {%.2f, %.2f}\n", #F, rx, ry, dx, dy);              \
  }

int main() {
  double x = 2, y = 3;
  TEST(issue, 4 * x * y, 2 * x * x);        // CHECK-EXEC: issue {24.00, 8.00} {24.00, 8.00}
  TEST(dependent, y * y, 2 * x * y);        // CHECK-EXEC: dependent {9.00, 12.00} {9.00, 12.00}
  TEST(through_this, 4 * x * y, 2 * x * x); // CHECK-EXEC: through_this {24.00, 8.00} {24.00, 8.00}
  TEST(member_read, y, x);                  // CHECK-EXEC: member_read {3.00, 2.00} {3.00, 2.00}
  TEST(copied, y, x);                       // CHECK-EXEC: copied {3.00, 2.00} {3.00, 2.00}
  TEST(copies, y, x);                       // CHECK-EXEC: copies {3.00, 2.00} {3.00, 2.00}
  TEST(assigned_object, y, x);              // CHECK-EXEC: assigned_object {3.00, 2.00} {3.00, 2.00}
  TEST(uninitialized, y, x);                // CHECK-EXEC: uninitialized {3.00, 2.00} {3.00, 2.00}
  TEST(nested, y, x);                       // CHECK-EXEC: nested {3.00, 2.00} {3.00, 2.00}
  TEST(functor, 3 * y, 3 * x);              // CHECK-EXEC: functor {9.00, 6.00} {9.00, 6.00}
  TEST(functor_of_x, y, x);                 // CHECK-EXEC: functor_of_x {3.00, 2.00} {3.00, 2.00}
  TEST(partial, y, x);                      // CHECK-EXEC: partial {3.00, 2.00} {3.00, 2.00}
  TEST(static_member, 5 * y, 5 * x);        // CHECK-EXEC: static_member {15.00, 10.00} {15.00, 10.00}
  TEST(global, 6 * x * y, 3 * x * x);       // CHECK-EXEC: global {36.00, 12.00} {36.00, 12.00}
  TEST(global_pointer, 6 * x * y, 3 * x * x); // CHECK-EXEC: global_pointer {36.00, 12.00} {36.00, 12.00}
  TEST(issue_2323, 2 * x * y, x * x);       // CHECK-EXEC: issue_2323 {12.00, 4.00} {12.00, 4.00}
  TEST(returned, y, x);                     // CHECK-EXEC: returned {3.00, 2.00} {3.00, 2.00}
  TEST(returned_call, y * y, 2 * x * y);    // CHECK-EXEC: returned_call {9.00, 12.00} {9.00, 12.00}
  TEST(seeded, 4 * y, 4 * x);               // CHECK-EXEC: seeded {12.00, 8.00} {12.00, 8.00}
  TEST(seeded_call, 8 * x * y, 4 * x * x);  // CHECK-EXEC: seeded_call {48.00, 16.00} {48.00, 16.00}
  TEST(mixed_branches, y, x);               // CHECK-EXEC: mixed_branches {3.00, 2.00} {3.00, 2.00}
  TEST(no_branch, 3 * y, 3 * x);            // CHECK-EXEC: no_branch {9.00, 6.00} {9.00, 6.00}
  TEST(from_lambda, y, x);                  // CHECK-EXEC: from_lambda {3.00, 2.00} {3.00, 2.00}
  TEST(by_value, y, x);                     // CHECK-EXEC: by_value {3.00, 2.00} {3.00, 2.00}
  TEST(zero, 2 * x + y, x);                 // CHECK-EXEC: zero {7.00, 2.00} {7.00, 2.00}
}
