// RUN: %cladclang %s -I%S/../../include -oReferenceArguments.out 2>&1 | %filecheck %s
// RUN: ./ReferenceArguments.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

double fn1(double& i, double& j) {
  double res = i * i * j;
  return res;
}

// CHECK: double fn1_darg0(double &i, double &j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double _t0 = i * i;
// CHECK-NEXT:     double _d_res = (_d_i * i + i * _d_i) * j + _t0 * _d_j;
// CHECK-NEXT:     double res = _t0 * j;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

// A function returning void leaves its result in a parameter. Its derivative
// returns the tangent of that parameter, which would otherwise be computed
// into a local the caller cannot read.
void fn2(double i, double j, double k, double& res) { res = i * j * k; }

// CHECK: double fn2_darg0(double i, double j, double k, double &res) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double _d_k = 0;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double _t0 = i * j;
// CHECK-NEXT:     _d_res = (_d_i * j + i * _d_j) * k + _t0 * _d_k;
// CHECK-NEXT:     res = _t0 * k;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

// A const reference is read, not written, so it is not the output: `res` is
// still the only one and the derivative still returns its tangent.
void fn3(double i, const double& c, double& res) { res = i * c; }

// CHECK: double fn3_darg0(double i, const double &c, double &res) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     const double _d_c = 0;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     _d_res = _d_i * c;
// CHECK-NEXT:     res = i * c;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

// With nothing written through a parameter there is no tangent to hand back,
// and the derivative keeps the primal's void return.
void fn4(double i) { double t = i * i; (void)t; }

// CHECK: void fn4_darg0(double i) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_t = _d_i * i + i * _d_i;
// CHECK-NEXT:     double t = i * i;
// CHECK-NEXT: }

// A static member function is addressed as a plain function pointer, so it
// takes the same path as a free function and gains the same return type. An
// instance method does not: its derivative is described through a member
// pointer type, which cannot carry one.
struct A {
  static void sf(double i, double& out) { out = i * i; }
};

// CHECK: static double sf_darg0(double i, double &out) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_out = 0;
// CHECK-NEXT:     _d_out = _d_i * i + i * _d_i;
// CHECK-NEXT:     out = i * i;
// CHECK-NEXT:     return _d_out;
// CHECK-NEXT: }

// An instance method keeps its void return: its derivative is described
// through a member pointer type, which cannot carry a changed one, and a
// signature the traits and the generated code disagree about is worse than a
// tangent reached through a wrapper.
struct B {
  double m = 3;
  void im(double i, double& out) { out = i * m; }
};

// CHECK: void im_darg0(double i, double &out) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_out = 0;
// CHECK-NEXT:     B _d_this_obj;
// CHECK-NEXT:     B *_d_this = &_d_this_obj;
// CHECK-NEXT:     double &_t0 = this->m;
// CHECK-NEXT:     _d_out = _d_i * _t0 + i * _d_this->m;
// CHECK-NEXT:     out = i * _t0;
// CHECK-NEXT: }

// A return that leaves the output alone has to leave the tangent alone with
// it. Dropping the statement would lose the path, not just the value: the
// derivative would run on and write what the primal did not.
void early(double x, double& out) {
  if (x < 0)
    return;
  out = x * x;
}

// CHECK: double early_darg0(double x, double &out) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_out = 0;
// CHECK-NEXT:     if (x < 0)
// CHECK-NEXT:         return _d_out;
// CHECK-NEXT:     _d_out = _d_x * x + x * _d_x;
// CHECK-NEXT:     out = x * x;
// CHECK-NEXT:     return _d_out;
// CHECK-NEXT: }

#define INIT(fn, ...) auto d_##fn = clad::differentiate(fn, __VA_ARGS__);

#define TEST(fn, ...)                                                          \
  auto res = d_##fn.execute(__VA_ARGS__);                                      \
  printf("{%.2f}\n", res)

int main() {
    INIT(fn1, "i");
    
    double i = 3, j = 5;
    TEST(fn1, i, j);    // CHECK-EXEC: {30.00}

    INIT(fn2, "i");
    double out = 0;
    printf("{%.2f}\n", d_fn2.execute(2, 3, 4, out));  // CHECK-EXEC: {12.00}
    printf("{%.2f}\n", out);                          // CHECK-EXEC: {24.00}

    INIT(fn3, "i");
    const double c = 5;
    out = 0;
    printf("{%.2f}\n", d_fn3.execute(2, c, out));     // CHECK-EXEC: {5.00}
    printf("{%.2f}\n", out);                          // CHECK-EXEC: {10.00}

    // Nothing to return, but it still differentiates and runs.
    INIT(fn4, "i");
    d_fn4.execute(3);

    auto d_sf = clad::differentiate(&A::sf, "i");
    out = 0;
    printf("{%.2f}\n", d_sf.execute(3, out));         // CHECK-EXEC: {6.00}
    printf("{%.2f}\n", out);                          // CHECK-EXEC: {9.00}

    // Still differentiates and runs; the tangent just stays inside.
    auto d_im = clad::differentiate(&B::im, "i");
    B b;
    out = 0;
    d_im.execute(b, 2, out);
    printf("{%.2f}\n", out);                          // CHECK-EXEC: {6.00}

    INIT(early, "x");
    // The taken path: the primal writes and the tangent follows.
    out = 0;
    printf("{%.2f}\n", d_early.execute(3, out));      // CHECK-EXEC: {6.00}
    printf("{%.2f}\n", out);                          // CHECK-EXEC: {9.00}
    // The early return: neither the output nor the tangent is touched.
    out = -1;
    printf("{%.2f}\n", d_early.execute(-2, out));     // CHECK-EXEC: {0.00}
    printf("{%.2f}\n", out);                          // CHECK-EXEC: {-1.00}
}
