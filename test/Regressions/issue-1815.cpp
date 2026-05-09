// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn -fsyntax-only -std=c++17 -I%S/../../include %s | FileCheck %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdarg>

typedef enum { a } b;
typedef enum { c, d } e;
e f;

int *g() {
  switch (f) { 
    case c: 
      break; 
  }
  return nullptr; 
}

void h(b, e, char[], char[], int, bool, char, char *, va_list) { 
  g(); 
}

char i, o, j;
int k;

void l(float val) {
  va_list arg;
  h(a, d, &i, &o, k, 0, '\0', &j, arg);
}

double n(double x) {
  l(x);
  return x * x; 
}

void check() {
  auto grad_n = clad::gradient(n);
}

// CHECK: void n_grad(double x, double *_d_x)
// CHECK: l(x);
// CHECK: *_d_x += 1 * x;
// CHECK: *_d_x += x * 1;