// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | %filecheck %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"


double fn_StmtExpr(double x, double y) {
  double val = ({ double z = y; x + z; }); // expected-warning {{statement kind 'StmtExpr' is not supported}} // expected-warning {{statement kind 'StmtExpr' is not supported}}
  return 0;
}

double id(double x) {
   return x;
}

double (*id_ptr)(double) = id; // expected-warning {{gradient uses a global variable 'id_ptr'; rerunning the gradient requires 'id_ptr' to be reset}}

double fn_indirect_call(double x, double y) {
  y = id_ptr(x); // expected-warning {{differentiation of indirect calls is not supported}}
  return 1;
}

int main() {
    clad::gradient(fn_StmtExpr);
    clad::differentiate(fn_StmtExpr, "x");
    clad::gradient<clad::opts::disable_tbr>(fn_indirect_call);
}
