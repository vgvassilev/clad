// RUN: %cladclang -fsyntax-only -Xclang -verify -std=c++17 -I%S/../../include %s
// RUN: %cladclang -fsyntax-only -Xclang -verify -std=c++17 -I%S/../../include %s 2>&1 | %filecheck %s --check-prefix=FAIL-CLOSED
// FAIL-CLOSED: void n_grad(float arg, float *_d_arg) {
// FAIL-CLOSED-NOT: <no independent variable specified>

#include "clad/Differentiator/Differentiator.h"

int* global_ptr; // expected-warning {{gradient uses a global variable}}

void use(int*) {}

double fn(double x) {
  use(global_ptr);
  return x * x;
}

void test() {
  auto grad_fn = clad::gradient(fn);
}

typedef enum { a } b;
typedef enum { c, d } e;
e f;

int *g() {
  switch (f) { 
    case c: break; 
    case d: break; 
  } 
} // expected-warning {{non-void function does not return a value}}
void h(b, e, char[], char[], int, bool, char, char *, va_list) { g(); }

char i, o, j; // expected-warning 3 {{gradient uses a global variable}}
int k; // expected-warning {{gradient uses a global variable}}

void l(float) {
  va_list arg;
  h(a, d, &i, &o, k, 0, '\0', &j, arg); 
}

float m; // expected-warning {{gradient uses a global variable}}

void n(float) {
  l(m);
}

void zero_param() {}

void test_nested_globals() {
  clad::gradient(n);
}

void test_zero_param_rejection() {
  clad::gradient(zero_param); // expected-error {{attempted to differentiate function with no parameters}}
}