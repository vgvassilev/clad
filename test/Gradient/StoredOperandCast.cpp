// RUN: %cladclang %s -I%S/../../include -fsyntax-only \
// RUN:   -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn-ast 2>&1 | FileCheck %s
// RUN: %cladclang %s -I%S/../../include -oStoredOperandCast.out
// RUN: ./StoredOperandCast.out | %filecheck_exec %s

// The reverse pass replaces an operand it saved with the saved value. An int
// read used as a double, `k * b.a`, keeps its conversion to double, and that
// conversion has to apply to the loaded value, not to the lvalue: a build of
// clang with assertions aborts on the latter.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct Ints { int a, b; };
void bump(Ints& s) { s.a += 1; }

double f(double k) {
  Ints b{1, 2};
  bump(b);
  return k * b.a;
}

// CHECK: ImplicitCastExpr{{.*}}col:{{[0-9]+}}{{.*}}'double'{{.*}}IntegralToFloating
// CHECK-NEXT: ImplicitCastExpr{{.*}}'int'{{.*}}LValueToRValue
// CHECK-NEXT: MemberExpr{{.*}}'int'{{.*}}lvalue{{.*}} .a

int main() {
  double dk = 0;
  clad::gradient(f).execute(3, &dk);
  printf("%.2f\n", dk); // CHECK-EXEC: 2.00
}
