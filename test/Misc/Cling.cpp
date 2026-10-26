// REQUIRES: cling
// %cling stands for a path and carries no flags, so the include paths are
// named here -- and clad's public headers expand a table that is rendered into
// the build directory, not committed beside them.
// RUN: cat %s | %cling -D__CLAD__ -I%S/../../include -I%clad_gen_incl 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not "clad generated code"

// Clad inside cling, asked for derivatives the way ROOT asks: a derivative
// it can build, one it cannot, and a session that carries on afterwards.

#include "clad/Differentiator/Differentiator.h"
extern "C" int printf(const char*, ...);
#pragma clad OFF

double h(double x) { return x * x * x; }
// Neither copyable nor default-constructible, so clad has no way to declare
// its adjoint and Sema rejects the declaration clad writes.
struct Uncopyable { double v; Uncopyable(double v) : v(v) {} Uncopyable(const Uncopyable&) = delete; };
double f(double x) { Uncopyable u(x); u.v = u.v * x; return u.v; }

#pragma clad ON
void h_req() { clad::gradient(h); }
#pragma clad OFF
double dh = 0;
h_grad(2, &dh);
printf("dh = %g\n", dh);
// CHECK: dh = 12

#pragma clad ON
void f_req() { clad::gradient(f); }
#pragma clad OFF
// The code the error is about is generated and has not been printed, so it
// has no line of its own to point at.
// CHECK: {{^}}error: no matching constructor for initialization of 'Uncopyable'
// CHECK-NEXT: {{^input_line_[0-9]+}}:1:{{[0-9]+}}: note: in the code clad generated for this statement
// CHECK-NEXT: double f(double x) {{.*}}
// CHECK-NEXT: {{^ +\^~+$}}
// CHECK-NEXT: {{^input_line_[0-9]+}}:1:16: note: in the derivative of 'f' requested here
// CHECK-NEXT: void f_req() { clad::gradient(f); }

printf("still running\n");
// CHECK: still running
