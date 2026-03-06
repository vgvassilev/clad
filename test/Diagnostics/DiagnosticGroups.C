// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify=default
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -plugin-arg-clad -Xclang -Wno-clad-unsupported -Xclang -verify=suppressed
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -plugin-arg-clad -Xclang -Wno-clad -Xclang -verify=off

#include "clad/Differentiator/Differentiator.h"

// suppressed-no-diagnostics
// off-no-diagnostics

double test_unsupported(double x) {
  double y = ({ double z = x; z; }); // default-warning {{statement kind 'StmtExpr' is not supported}}
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
