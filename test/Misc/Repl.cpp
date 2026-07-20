// REQUIRES: clang-repl
// UNSUPPORTED: clang-13, clang-14, clang-15, clang-16, clang-17, clang-18
// RUN: cat %s | %clang-repl -Xcc -fplugin=%cladlib                            \
// RUN:             -Xcc -I%S/../../include -Xcc -Xclang -Xcc -verify |        \
// RUN:                FileCheck %s

double sq(double x) { return x*x; }
extern "C" int printf(const char*,...);
#include "clad/Differentiator/Differentiator.h"
auto dsq = clad::differentiate(sq, "x");
auto r1 = printf("dsq(1)=%f\n", dsq.execute(1));
// CHECK: dsq(1)=2.00

int x = 5;
%undo
int y = x; // expected-error {{use of undeclared identifier 'x'}}
int x = 10;
%quit
