// REQUIRES: clang-repl
// UNSUPPORTED: clang-13, clang-14, clang-15, clang-17
// RUN: cat %s | %clang-repl -Xcc -fplugin=%cladlib -Xcc -I%S/../../include | FileCheck %s

double sq(double x) { return x*x; }
extern "C" int printf(const char*,...);
#include "clad/Differentiator/Differentiator.h"
auto dsq = clad::differentiate(sq, "x");
auto r1 = printf("dsq(1)=%f\n", dsq.execute(1));
// CHECK: dsq(1)=2.00
