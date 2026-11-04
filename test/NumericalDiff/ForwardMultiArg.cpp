// Forward mode falls back for a function of several arguments too: the
// tangent of the call is each partial derivative times its argument's
// tangent.
// RUN: %cladnumdiffclang %s -I%S/../../include -oForwardMultiArg.out \
// RUN:   -Wno-braced-scalar-init -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./ForwardMultiArg.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

// expected-warning@+2 {{attempted differentiation of function 'copysign'}}
// expected-note@+1 {{falling back to numerical differentiation for 'copysign'}}
double f(double x, double y) { return std::copysign(x * x, y + x); }

// CHECK: double f_darg0(double x, double y) {
// CHECK: return numerical_diff::forward_central_difference_tangent<double (*)(double, double){{.*}}>(std::copysign, _d_x * x + x * _d_x, x * x, 0, 0, x * x, y + x) + numerical_diff::forward_central_difference_tangent<double (*)(double, double){{.*}}>(std::copysign, _d_y + _d_x, y + x, 1, 0, x * x, y + x);

// The constant has no tangent, so it gets no partial: stepping it would only
// add evaluations, or reach where the function is not defined. (copysign was
// reported for f.)
double g(double x) { return std::copysign(x * x, 2.0); }

// CHECK: double g_darg0(double x) {
// CHECK-NEXT: double _d_x = 1;
// CHECK-NEXT: return numerical_diff::forward_central_difference_tangent<double (*)(double, double){{.*}}>(std::copysign, _d_x * x + x * _d_x, x * x, 0, 0, x * x, 2.);

// An enumeration takes no step, and has no tangent to multiply.
#define STR(x) #x
#define LABEL(prefix, name) STR(prefix) #name
enum Scale { Twice = 2 };
double scaled(double x, Scale s) __asm__(LABEL(__USER_LABEL_PREFIX__, scaled_impl));
extern "C" double scaled_impl(double x, Scale s) { return s * x * x; }
// expected-warning@+2 {{attempted differentiation of function 'scaled'}}
// expected-note@+1 {{falling back to numerical differentiation for 'scaled'}}
double h(double x) { return scaled(x, Twice); }

// An integer cannot take a step smaller than one, so it gets no partial.
double scaledBy(double x, int n) __asm__(LABEL(__USER_LABEL_PREFIX__, scaledBy_impl));
extern "C" double scaledBy_impl(double x, int n) { return n * x * x; }
// expected-warning@+2 {{attempted differentiation of function 'scaledBy'}}
// expected-note@+1 {{falling back to numerical differentiation for 'scaledBy'}}
double k(double x) { return scaledBy(x, static_cast<int>(x)); }

// x * sqrt(y) + x: at y = 0 a step in y leaves where sqrt is defined.
double xsqrty(double x, double y) __asm__(LABEL(__USER_LABEL_PREFIX__, xsqrty_impl));
extern "C" double xsqrty_impl(double x, double y) { return x * std::sqrt(y) + x; }
// A tangent that is zero only when the code runs -- y, not differentiated; c,
// held at a constant -- takes no step either.
// expected-warning@+2 {{attempted differentiation of function 'xsqrty'}}
// expected-note@+1 {{falling back to numerical differentiation for 'xsqrty'}}
double heldParam(double x, double y) { return xsqrty(x, y); }
double heldLocal(double x) {
  double c = 0;
  return xsqrty(x, c);
}
// In vector mode too, with a constant's tangent a vector of zeros. (A
// derivative of its own, so xsqrty is reported for it again.)
// expected-warning@+2 {{attempted differentiation of function 'xsqrty'}}
// expected-note@+1 {{falling back to numerical differentiation for 'xsqrty'}}
double vecConst(double x, double y) { return xsqrty(x, 0.0) + y; }
double vecLocal(double x, double y) {
  double c = 0;
  return xsqrty(x, c) + y;
}

double mul2(double a, double b) __asm__(LABEL(__USER_LABEL_PREFIX__, mul2_impl));
extern "C" double mul2_impl(double a, double b) { return a * b; }
// A tangent with an effect is evaluated once, as the argument is, and
// before it, as for a statement: it reads what its argument changes. In the
// sum, another term reading what it changes would be unsequenced with it.
// expected-warning@+2 {{attempted differentiation of function 'mul2'}}
// expected-note@+1 {{falling back to numerical differentiation for 'mul2'}}
double effects(double x, double y) { return mul2(y += x, x * x); }

// CHECK: double effects_darg0(double x, double y) {
// CHECK: double _t0 = _d_y += _d_x;
// CHECK-NEXT: double _t1 = y += x;
// CHECK-NEXT: double _t2 = _d_x * x + x * _d_x;
// CHECK-NEXT: double _t3 = x * x;
// CHECK-NEXT: return numerical_diff::forward_central_difference_tangent<double (*)(double, double){{.*}}>(mul2, _t0, _t1, 0, 0, _t1, _t3) + numerical_diff::forward_central_difference_tangent<double (*)(double, double){{.*}}>(mul2, _t2, _t3, 1, 0, _t1, _t3);

// y * x: the tangent reads y before its argument changes it.
double updates(double x, double y) { return mul2(y *= x, 1.0); }
// In vector mode a tangent with an effect is a call to an operator. (A
// derivative of its own, so mul2 is reported for it again.)
// expected-warning@+2 {{attempted differentiation of function 'mul2'}}
// expected-note@+1 {{falling back to numerical differentiation for 'mul2'}}
double vecEffects(double x, double y) { return mul2(y += x, x * x); }
// A tangent that is an expression of vectors, with one argument or several.
double cube(double a) __asm__(LABEL(__USER_LABEL_PREFIX__, cube_impl));
extern "C" double cube_impl(double a) { return a * a * a; }
// expected-warning@+2 {{attempted differentiation of function 'cube'}}
// expected-note@+1 {{falling back to numerical differentiation for 'cube'}}
double vecExprs(double x, double y) { return cube(x * y) + mul2(x * y, y); }


// An integer passed to a double is stepped as the double it is there.
double intPassed(double x, double y) {
  int n = x;
  return mul2(y, n);
}

// An argument that calls something is evaluated once, as the call has it.
double nextv() __asm__(LABEL(__USER_LABEL_PREFIX__, nextv_impl));
extern "C" {
int nextvCalls = 0;
double nextv_impl() { return ++nextvCalls; }
}
double callArg(double x) { return mul2(x, nextv()); }

// A hoisted vector tangent is kept as a vector, not as an expression of
// vectors that refers to temporaries.
double vecHoisted(double x, double y, double z) {
  return mul2((y += x) * x, z += 1.0);
}

// In vector mode, an integer's tangents do not truncate the partial.
double vecIntPassed(double x, double y) {
  int n = x;
  return mul2(y, n);
}

// The arguments and their tangents are evaluated in order: x * G is read
// before bump() changes G.
double G = 1;
double bump() __asm__(LABEL(__USER_LABEL_PREFIX__, bump_impl));
extern "C" double bump_impl() { return G += 10, 1; }
double inOrder(double x) { return mul2(x * G, bump()); }

// In vector mode the terms' sum is a vector, not an expression of the vectors
// the terms give: *= stores it, and reads it once they are gone.
double vecScaled(double x, double y) {
  double r = y;
  r *= mul2(x, y);
  return r;
}
// Of integers alone, a vector of doubles.
double vecIntArgs(double x, double y) {
  int n = x, m = y;
  return mul2(n, m);
}
// *= stores an expression of vectors as a vector of the wider floating type
// of the two sides' tangents, or of doubles where neither is floating.
double vecMulAssign(double x, double y) {
  double t = x;
  float f = y;
  int n = y, k = x;
  t *= f * f;
  t *= n * 2;
  k *= n * 2;
  return t + k;
}

// A float's tangent stays a float: a callee's pushforward pairs the two.
float fmul2(float a, float b) __asm__(LABEL(__USER_LABEL_PREFIX__, fmul2_impl));
extern "C" float fmul2_impl(float a, float b) { return a * b; }
// expected-warning@+2 {{attempted differentiation of function 'fmul2'}}
// expected-note@+1 {{falling back to numerical differentiation for 'fmul2'}}
float fcallee(float x, float y) { return fmul2(x, y); }
float viaFloat(float x, float y) { return fcallee(x, y) * 2; }
// Given a double tangent too: the call's tangent is a float, as the call is.
float mixedCallee(float x, double y) { return fmul2(x, y); }
float viaMixed(float x, double y) { return mixedCallee(x, y) * 2; }
// In vector mode a vector of floats, beside a float's in a ?:. (A derivative
// of its own, so fmul2 is reported for it again.)
// expected-warning@+2 {{attempted differentiation of function 'fmul2'}}
// expected-note@+1 {{falling back to numerical differentiation for 'fmul2'}}
float floatBranch(float x, float y) { return x > 0 ? fmul2(x, y) : x; }

// Arguments past the parameters cannot take a step.
double vsum(double x, double y, ...) __asm__(LABEL(__USER_LABEL_PREFIX__, vsum_impl));
extern "C" double vsum_impl(double x, double y, ...) { return x * y; }
// expected-warning@+3 {{attempted differentiation of function 'vsum'}}
// expected-note@+2 {{falling back to numerical differentiation for 'vsum'}}
// expected-warning@+1 {{numerical differentiation of 'vsum' is not viable for this call: it passes arguments past the parameters; considering its derivative as 0}}
double variadic(double x, double y) { return vsum(x, y, 3.0); }

// A braced argument cannot be evaluated beforehand, as the others are then.
// expected-warning@+1 {{numerical differentiation of 'mul2' is not supported here: its arguments have side effects, so they are evaluated beforehand, which a braced one cannot be; considering its derivative as 0}}
double braced(double x, double y) { return mul2({x}, y += x); }

// The derivative of a numerically differentiated call is not taken again.
// expected-warning@+3 {{attempted differentiation of function 'cube'}}
// expected-note@+2 {{falling back to numerical differentiation for 'cube'}}
// expected-warning@+1 {{the derivative of a numerically differentiated call is not computed; considering it as 0}}
double cubed(double x) { return cube(x); }

int main() {
  // copysign(x*x, y+x) is x*x for y + x > 0, so its derivative is 2x; in
  // the sign the second argument only sets, the partial is zero.
  auto d = clad::differentiate(f, "x");
  printf("%.2f\n", d.execute(3, 1)); // CHECK-EXEC: 6.00

  printf("%.2f\n", clad::differentiate(g, "x").execute(3)); // CHECK-EXEC: 6.00
  // 2 * x * x, at x = 3.
  printf("%.2f\n", clad::differentiate(h, "x").execute(3)); // CHECK-EXEC: 12.00
  // n * x * x with n = 3 held: 2 * n * x at x = 3.
  printf("%.2f\n", clad::differentiate(k, "x").execute(3)); // CHECK-EXEC: 18.00

  // sqrt(y) + 1 at y = 0.
  printf("%.2f\n", clad::differentiate(heldParam, "x").execute(2, 0)); // CHECK-EXEC: 1.00
  printf("%.2f\n", clad::differentiate(heldLocal, "x").execute(2)); // CHECK-EXEC: 1.00
  double dx = 0, dy = 0;
  clad::differentiate<clad::opts::vector_mode>(vecConst, "x,y").execute(2, 3, &dx, &dy);
  printf("{%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: {1.00, 1.00}
  clad::differentiate<clad::opts::vector_mode>(vecLocal, "x,y").execute(2, 3, &dx, &dy);
  printf("{%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: {1.00, 1.00}

  // (y + x) * x * x: x * x + 2 x (y + x) at (1, 2).
  printf("%.2f\n", clad::differentiate(effects, "x").execute(1, 2)); // CHECK-EXEC: 7.00
  // y at (2, 3).
  printf("%.2f\n", clad::differentiate(updates, "x").execute(2, 3)); // CHECK-EXEC: 3.00
  // (y + x) x^2 at (2, 3): {x^2 + 2 x (y + x), x^2}.
  clad::differentiate<clad::opts::vector_mode>(vecEffects, "x,y").execute(2, 3, &dx, &dy);
  printf("{%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: {24.00, 4.00}
  // (x y)^3 + x y^2 at (2, 3): {3 x^2 y^3 + y^2, 3 x^3 y^2 + 2 x y}.
  clad::differentiate<clad::opts::vector_mode>(vecExprs, "x,y").execute(2, 3, &dx, &dy);
  printf("{%.0f, %.0f}\n", dx, dy); // CHECK-EXEC: {333, 228}

  // y * n, with n's tangent that of x.
  printf("%.2f\n", clad::differentiate(intPassed, "x").execute(2, 2.5)); // CHECK-EXEC: 2.50
  // x * nextv(): the first value, from the one call.
  double dcall = clad::differentiate(callArg, "x").execute(2);
  printf("%.2f %d\n", dcall, nextvCalls); // CHECK-EXEC: 1.00 1
  // (y + x) x (z + 1) at (2, 3, 4): {(2 x + y)(z + 1), x (z + 1), (y + x) x}.
  double dz = 0;
  clad::differentiate<clad::opts::vector_mode>(vecHoisted, "x,y,z").execute(2, 3, 4, &dx, &dy, &dz);
  printf("{%.0f, %.0f, %.0f}\n", dx, dy, dz); // CHECK-EXEC: {35, 10, 10}
  // y n at (3, 2.5): {y, n}.
  clad::differentiate<clad::opts::vector_mode>(vecIntPassed, "x,y").execute(3, 2.5, &dx, &dy);
  printf("{%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: {2.50, 3.00}
  // G, as it was when x * G was evaluated.
  printf("%.2f\n", clad::differentiate(inOrder, "x").execute(3)); // CHECK-EXEC: 1.00
  // x y^2 at (3, 2): {y^2, 2 x y}.
  clad::differentiate<clad::opts::vector_mode>(vecScaled, "x,y").execute(3, 2, &dx, &dy);
  printf("{%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: {4.00, 12.00}
  // n m at (2, 3): {m, n}.
  clad::differentiate<clad::opts::vector_mode>(vecIntArgs, "x,y").execute(2, 3, &dx, &dy);
  printf("{%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: {3.00, 2.00}
  // 2 x y^3 + 2 x y at (2, 3): {2 y^3 + 2 y, 6 x y^2 + 2 x}.
  clad::differentiate<clad::opts::vector_mode>(vecMulAssign, "x,y").execute(2, 3, &dx, &dy);
  printf("{%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: {60.00, 112.00}
  // 2 x y at (3, 2): 2 y.
  printf("%.2f\n", clad::differentiate(viaFloat, "x").execute(3, 2)); // CHECK-EXEC: 4.00
  printf("%.2f\n", clad::differentiate(viaMixed, "x").execute(3, 2)); // CHECK-EXEC: 4.00
  float fx = 0, fy = 0;
  clad::differentiate<clad::opts::vector_mode>(floatBranch, "x,y").execute(3, 2, &fx, &fy);
  printf("{%.2f, %.2f}\n", fx, fy); // CHECK-EXEC: {2.00, 3.00}
  printf("%.2f\n", clad::differentiate(variadic, "x").execute(2, 3)); // CHECK-EXEC: 0.00
  // x (y + x) would give 2 x + y; the call's term is dropped.
  printf("%.2f\n", clad::differentiate(braced, "x").execute(2, 3)); // CHECK-EXEC: 0.00
  // 6 x would need the second derivative of cube.
  printf("%.2f\n", clad::differentiate<2>(cubed, "x").execute(2)); // CHECK-EXEC: 0.00
}
