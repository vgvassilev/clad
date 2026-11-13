// RUN: %cladclang -fsyntax-only -Xclang -verify -std=c++17 -I%S/../../include %s

// Missing tangents at a return statement diagnose cleanly when the return type
// has no zero value or is a non-const lvalue reference.

#include "clad/Differentiator/Differentiator.h"

struct ND {
  double a;
  ND(double x) : a(x) {}
};

double g = 1.0;

// expected-warning@+2 3 {{statement kind 'StmtExpr' is not supported}}
// expected-error@+1 3 {{cannot differentiate 'f_nd': the returned expression has no derivative and 'ND' has no zero value}}
ND f_nd(double x) { return ({ ND n(x); n; }); }

// expected-warning@+2 3 {{statement kind 'StmtExpr' is not supported}}
// expected-error@+1 3 {{cannot differentiate 'f_ref_sub': the returned reference refers to a value without a derivative}}
double& f_ref_sub(double x) { return ({ &g; })[0]; }

// expected-error@+1 3 {{cannot differentiate 'f_ref_global': the returned reference refers to a value without a derivative}}
double& f_ref_global(double x) { return g; }

double h_nd(double x) { return f_nd(x).a; }
double h_nd_vec(double x) { f_nd(x); return x; }
double h_ref_sub(double x) { return f_ref_sub(x); }
double h_ref_global(double x) { return f_ref_global(x); }

int main() {
  clad::differentiate(f_nd, "x");
  clad::differentiate(f_ref_sub, "x");
  clad::differentiate(f_ref_global, "x");
  clad::differentiate(h_nd, "x");
  clad::differentiate(h_ref_sub, "x");
  clad::differentiate(h_ref_global, "x");
  clad::differentiate<clad::opts::vector_mode>(h_nd_vec);
  clad::differentiate<clad::opts::vector_mode>(h_ref_sub);
  clad::differentiate<clad::opts::vector_mode>(h_ref_global);
}
