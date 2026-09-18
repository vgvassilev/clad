// REQUIRES: cling
// RUN: cat %s | %cling -D__CLAD__ -I%S/../../include 2>&1 | FileCheck %s

// Clad inside cling, asked for derivatives the way ROOT asks: a derivative
// it can build, one it cannot, and a session that carries on afterwards.

#include "clad/Differentiator/Differentiator.h"
extern "C" int printf(const char*, ...);
#pragma clad OFF

double h(double x) { return x * x * x; }
struct B { double v; };
struct D : B {};
double f(double x) { D d; d.v = x; B* b = &d; D* p = static_cast<D*>(b); return p->v; }

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
// CHECK: error: cannot initialize a variable of type 'D *' with an lvalue of type 'B *'

printf("still running\n");
// CHECK: still running
