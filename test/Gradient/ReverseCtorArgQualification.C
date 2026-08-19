// RUN: %cladclang %s -I%S/../../include -oReverseCtorArgQualification.out
// RUN: ./ReverseCtorArgQualification.out | %filecheck_exec %s
//
// Regression test for the plugin availability of
// clang::TypeName::getFullyQualifiedType. It is used by
// CladUtils::makeTypeReadable in reverse-mode call-argument handling, but within
// clang only the Interpreter library references it. When clad is loaded as a
// -fplugin into a clang driver built without LLVM_LINK_LLVM_DYLIB, the symbol is
// absent from the host binary; without clad supplying it itself
// (tools/RequiredSymbols.cpp + clangAST archive linkage) the call binds to null
// and reverse-differentiating any function that constructs a class object
// crashes with a jump to address 0. See issue vgvassilev/clad#XXXX.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct Pair {
  double a, b;
  Pair(double x, double y) : a(x), b(y) {}
};

double f(double x) {
  Pair p(x * x, x);
  return p.a + p.b; // x^2 + x  ->  f'(x) = 2x + 1
}

int main() {
  auto g = clad::gradient(f);
  double dx = 0;
  g.execute(3.0, &dx);
  printf("dx = %.2f\n", dx); // CHECK-EXEC: dx = 7.00
  return 0;
}
