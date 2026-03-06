// RUN: %cladclang %s -I%S/../../include -fsyntax-only 2>&1 | %filecheck %s --check-prefix=CHECK-DEFAULT
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -plugin-arg-clad -Xclang -Wno-clad-unsupported 2>&1 | %filecheck %s --check-prefix=CHECK-NO-UNSUPPORTED
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -plugin-arg-clad -Xclang -Wno-clad 2>&1 | %filecheck %s --check-prefix=CHECK-NO-CLAD

#include "clad/Differentiator/Differentiator.h"

double id(double x) { return x; }
double (*id_ptr)(double) = id;

double test_unsupported(double x) {
  // CHECK-DEFAULT: indirect calls
  // CHECK-DEFAULT: statement kind 'StmtExpr' is not supported
  // CHECK-NO-UNSUPPORTED-NOT: indirect calls
  // CHECK-NO-UNSUPPORTED-NOT: statement kind 'StmtExpr' is not supported
  // CHECK-NO-CLAD-NOT: indirect calls
  // CHECK-NO-CLAD-NOT: statement kind 'StmtExpr' is not supported
  double y = ({ double z = x; z; });
  y += id_ptr(x);
  return y;
}

double test_valid(double x) {
  return x * x + 2 * x + 1;
}

int main() {
  auto grad = clad::gradient(test_valid);
  auto bad = clad::gradient(test_unsupported);
  (void)grad;
  (void)bad;
  return 0;
}
