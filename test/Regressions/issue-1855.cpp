// RUN: %cladclang -fsyntax-only -Xclang -verify -std=c++17 -I%S/../../include %s

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

void n() {
  l(m);
  clad::gradient(n); // expected-error {{attempted to differentiate function with no parameters}}
}