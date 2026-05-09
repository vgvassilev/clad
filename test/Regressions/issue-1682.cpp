// RUN: %cladclang -fsyntax-only -Xclang -verify %s -I%S/../../include

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);
void a(...);
int b; // expected-warning {{gradient uses a global variable 'b'; rerunning the gradient requires 'b' to be reset}}

void c(float) {
    // expected-warning@+2 {{attempted differentiation of function 'a' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
    // expected-note@+1 {{numerical differentiation is not viable for 'a'; considering 'a' as 0}}
    a(b);
}

double f0(double x) {
    int index = 10 % 3;
    const char* func = __func__;
    void* ptr = __null;

    // expected-warning@+2 {{attempted differentiation of function 'printf' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
    // expected-note@+1 {{numerical differentiation is not viable for 'printf'; considering 'printf' as 0}}
    printf("%f", x);

    return x * x;
}

auto df = clad::gradient(f0);
auto dg = clad::gradient(c);

int b_fwd;

void c_fwd(float) {
    // expected-warning@+2 {{attempted differentiation of function 'a' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
    // expected-note@+1 {{numerical differentiation is not viable for 'a'; considering 'a' as 0}}
    a(b_fwd);
}

double f0_fwd(double x) {
    int index = 10 % 3;
    const char* func = __func__;
    void* ptr = __null;

    // expected-warning@+2 {{attempted differentiation of function 'printf' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
    // expected-note@+1 {{numerical differentiation is not viable for 'printf'; considering 'printf' as 0}}
    printf("%f", x);

    return x * x;
}

auto df_fwd = clad::differentiate(f0_fwd);
auto dg_fwd = clad::differentiate(c_fwd);