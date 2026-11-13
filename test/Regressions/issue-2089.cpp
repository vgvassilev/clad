// RUN: %cladclang -std=c++17 %s -I%S/../../include -Xclang -verify -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// Missing tangents (such as unsupported StmtExpr or indirect calls) leave
// Expr_dx as null. Returning or combining them must emit valid C++, clone local
// declarations inside the unsupported statement without stale references, and
// compute the surviving derivative terms in scalar, pushforward, and vector
// modes.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct Point {
  double a;
};

struct DefCtor {
  double a;
  DefCtor() : a(0) {}
  DefCtor(double v) : a(v) {}
};

double g = 1.0;

// expected-warning@+1 3 {{statement kind 'StmtExpr' is not supported}}
double f_stmt(double x) { return ({ double t = x * x; t + 1; }); }
// CHECK: double f_stmt_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     return 0.;
// CHECK-NEXT: }

// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
auto f_auto(double x) { return ({ double t = x * x; t + 1; }); }

// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
Point f_struct(double x) { return ({ Point p{x}; p; }); }

// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
DefCtor f_defctor(double x) { return ({ DefCtor d(x); d; }); }

// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
double* f_ptr(double x) { return ({ &g; }); }

// expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
double helper_pf(double x) { return ({ double t = x * x; t + 1; }); }
double f_push(double x) { return x + 2.0 * helper_pf(x); }

// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
Point helper_pf_pt(double x) { return ({ Point p{x}; p; }); }
double f_push_pt(double x) { return x + helper_pf_pt(x).a; }

// expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
void helper_void_pf(double x) { return (void)({ double t = x; t; }); }
double f_void_pf(double x) {
  helper_void_pf(x);
  return x * x;
}

// expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
void helper_void_comma(double x, double& out) { return (out = x, (void)({ double t = x; t; })); }
double f_void_comma(double x) {
  double out = 0;
  helper_void_comma(x, out);
  return out;
}

// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
double f_add1(double x) { return x + ({ double t = x * x; t + 1; }); }
// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
double f_add2(double x) { return ({ double t = x * x; t + 1; }) + x; }
// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
double f_sub1(double x) { return x - ({ double t = x * x; t + 1; }); }
// expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
double f_sub2(double x) { return ({ double t = x * x; t + 1; }) - 2.0 * x; }

// expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
double f_div1(double x) { return ({ double t = x * x; t + 1; }) / x; }
// CHECK: double f_div1_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _t0 = ({
// CHECK-NEXT:         double t = x * x;
// CHECK-NEXT:         t + 1;
// CHECK-NEXT:     });
// CHECK-NEXT:     return -(_t0 * _d_x) / (x * x);
// CHECK-NEXT: }

// Homonymous outer variable `t` must not capture `t` inside cloned StmtExpr.
double f_div_shadow(double x) {
  double t = 100.0;
  // expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
  double r = ({ double t = x * x; t + 1; }) / x;
  return r + 0.0 * t;
}

// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
double f_div2(double x) { return x / ({ double t = x * x; t + 1; }); }
// expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
double f_mul0(double x) { return ({ double t = x; t; }) * ({ double t = x; t + 1; }); }

double f_assign(double x) {
  double y = x;
  // expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
  y = ({ double t = x * x; t + 1; });
  return y;
}

double f_mulassign(double x) {
  double y = x;
  // expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
  y *= ({ double t = x * x; t + 1; });
  return y;
}

double f_divassign(double x) {
  double y = x;
  // expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
  y /= ({ double t = x * x; t + 1; });
  return y;
}

double f_comma(double x) {
  double out = 0;
  // expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
  return (out = x, (void)({ double t = x; t; })), out;
}

void f_void(double x, double& out) {
  out = x;
  // expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
  return (void)({ double t = x; t; });
}

void f_void_early(double x, double& out) {
  if (x < 0)
    return;
  out = x;
  // expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
  return (void)({ double t = x; t; });
}

// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
double f_ccast(double x) { return (double)({ double t = x * x; t + 1; }); }
// expected-warning@+1 {{statement kind 'StmtExpr' is not supported}}
double f_fcast(double x) { return double(({ double t = x * x; t + 1; })); }
// expected-warning@+1 2 {{statement kind 'StmtExpr' is not supported}}
double f_cond(double x) { return x > 0 ? x : ({ double t = x; t; }); }

// CHECK: void f_cond_dvec(double x, double *_d_x) {
// CHECK:     clad::array<double> _d_vector_return(_t0 ? _d_vector_x : clad::zero_vector(indepVarCount));
// CHECK: }

// CHECK: void f_void_dvec(double x, double &out, double *_d_x, double *_d_out) {
// CHECK:     _d_vector_out = _d_vector_x;
// CHECK:     out = x;
// CHECK:     return;
// CHECK: }

int main() {
  printf("%.2f %.2f %.2f %.2f %d\n",
         clad::differentiate(f_stmt, "x").execute(3.0),
         clad::differentiate(f_auto, "x").execute(3.0),
         clad::differentiate(f_struct, "x").execute(3.0).a,
         clad::differentiate(f_defctor, "x").execute(3.0).a,
         clad::differentiate(f_ptr, "x").execute(3.0) == nullptr);
  // CHECK-EXEC: 0.00 0.00 0.00 0.00 1

  printf("%.2f %.2f %.2f %.2f\n",
         clad::differentiate(f_push, "x").execute(3.0),
         clad::differentiate(f_push_pt, "x").execute(3.0),
         clad::differentiate(f_void_pf, "x").execute(3.0),
         clad::differentiate(f_void_comma, "x").execute(3.0));
  // CHECK-EXEC: 1.00 1.00 6.00 1.00

  printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
         clad::differentiate(f_add1, "x").execute(3.0),
         clad::differentiate(f_add2, "x").execute(3.0),
         clad::differentiate(f_sub1, "x").execute(3.0),
         clad::differentiate(f_sub2, "x").execute(3.0),
         clad::differentiate(f_div1, "x").execute(2.0),
         clad::differentiate(f_div_shadow, "x").execute(2.0),
         clad::differentiate(f_div2, "x").execute(2.0),
         clad::differentiate(f_mul0, "x").execute(3.0));
  // CHECK-EXEC: 1.00 1.00 1.00 -2.00 -1.25 -1.25 0.20 0.00

  double out = 0;
  printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
         clad::differentiate(f_assign, "x").execute(3.0),
         clad::differentiate(f_mulassign, "x").execute(2.0),
         clad::differentiate(f_divassign, "x").execute(2.0),
         clad::differentiate(f_comma, "x").execute(3.0),
         clad::differentiate(f_void, "x").execute(3.0, out),
         clad::differentiate(f_void_early, "x").execute(-3.0, out),
         clad::differentiate(f_void_early, "x").execute(3.0, out),
         clad::differentiate(f_ccast, "x").execute(3.0),
         clad::differentiate(f_fcast, "x").execute(3.0),
         clad::differentiate(f_cond, "x").execute(3.0),
         clad::differentiate(f_cond, "x").execute(-3.0));
  // CHECK-EXEC: 0.00 5.00 0.20 1.00 1.00 0.00 1.00 0.00 0.00 1.00 0.00

  double dx = 0, dout = 0;
  clad::differentiate<clad::opts::vector_mode>(f_stmt).execute(3.0, &dx);
  printf("%.2f ", dx);
  clad::differentiate<clad::opts::vector_mode>(f_sub2).execute(3.0, &dx);
  printf("%.2f ", dx);
  clad::differentiate<clad::opts::vector_mode>(f_div1).execute(2.0, &dx);
  printf("%.2f ", dx);
  clad::differentiate<clad::opts::vector_mode>(f_div_shadow).execute(2.0, &dx);
  printf("%.2f ", dx);
  clad::differentiate<clad::opts::vector_mode>(f_assign).execute(3.0, &dx);
  printf("%.2f ", dx);
  clad::differentiate<clad::opts::vector_mode>(f_mulassign).execute(2.0, &dx);
  printf("%.2f ", dx);
  clad::differentiate<clad::opts::vector_mode>(f_push).execute(3.0, &dx);
  printf("%.2f ", dx);
  clad::differentiate<clad::opts::vector_mode>(f_cond).execute(3.0, &dx);
  printf("%.2f ", dx);
  clad::differentiate<clad::opts::vector_mode>(f_cond).execute(-3.0, &dx);
  printf("%.2f\n", dx);
  // CHECK-EXEC: 0.00 -2.00 -1.25 -1.25 0.00 5.00 1.00 1.00 0.00

  dx = 0; dout = 0;
  clad::differentiate<clad::opts::vector_mode>(f_void).execute(3.0, out, &dx, &dout);
  printf("%.2f %.2f ", dx, dout);
  dx = 0; dout = 0;
  clad::differentiate<clad::opts::vector_mode>(f_void_early).execute(-3.0, out, &dx, &dout);
  printf("%.2f %.2f ", dx, dout);
  dx = 0;
  clad::differentiate<clad::opts::vector_mode>(f_void_pf).execute(3.0, &dx);
  printf("%.2f ", dx);
  dx = 0;
  clad::differentiate<clad::opts::vector_mode>(f_void_comma).execute(3.0, &dx);
  printf("%.2f\n", dx);
  // CHECK-EXEC: 0.00 0.00 0.00 0.00 6.00 1.00

  dx = 0;
  clad::gradient(f_stmt).execute(3.0, &dx);
  printf("%.2f ", dx);
  dx = 0;
  clad::gradient(f_push).execute(3.0, &dx);
  printf("%.2f\n", dx);
  // CHECK-EXEC: 0.00 1.00
}
