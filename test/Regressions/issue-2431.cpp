// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -o %t
// RUN: %t | %filecheck_exec %s

// Reverse mode moves a local declared in a nested scope to function scope.
// Where it would assign a new value to the moved variable, it constructs a new
// object in its storage instead when the type cannot be assigned (#1217,
// #2431).

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct NonAssignable {
  double x;
  NonAssignable operator=(const NonAssignable&) = delete;
};

double used(double x) {
  double r = 0;
  for (int i = 0; i < 2; ++i) {
    NonAssignable u{x};
    r += u.x;
  }
  return r;
}

// CHECK-LABEL: void used_grad(double x, double *_d_x) {
// CHECK: NonAssignable _d_u = {0.};
// CHECK: NonAssignable u = {0.};
// CHECK: ::new (static_cast<void *>(clad_addressof(u))) NonAssignable{x};
// CHECK: ::new (static_cast<void *>(clad_addressof(_d_u))) NonAssignable{0.};
// CHECK: }

// Two declarators in one statement, one of them never read (#1217).
double declarators(double x) {
  double r = 0;
  for (int i = 0; i < 2; ++i) {
    NonAssignable u{x}, v{2 * x}, w;
    r += u.x + v.x;
  }
  return r;
}

// The reverse sweep reads a different value of the variable in each
// iteration, so it is stored and restored without an assignment.
double distinct(double x) {
  double r = 0;
  for (int i = 0; i < 3; ++i) {
    NonAssignable u{x + i};
    r += u.x * u.x;
  }
  return r;
}

// CHECK-LABEL: void distinct_grad(double x, double *_d_x) {
// CHECK: clad::tape<NonAssignable> [[T:_t[0-9]+]] = {};
// CHECK: clad::push([[T]], std::move(u)) , ::new (static_cast<void *>(clad_addressof(u))) NonAssignable{x + i};
// CHECK: clad::reconstruct(u, clad::pop([[T]]));
// CHECK: }

// Only the value of the variable is needed in the reverse sweep.
double inactive(double x) {
  double r = 0;
  for (int i = 0; i < 3; ++i) {
    NonAssignable c{static_cast<double>(i)};
    r += c.x * x;
  }
  return r;
}

void square(NonAssignable& value) { value.x *= value.x; }

// A call that changes the object through a reference, here reached through a
// pointer to it.
double by_reference(double x) {
  double r = 0;
  for (int i = 0; i < 3; ++i) {
    NonAssignable u{x + i};
    NonAssignable* p = &u;
    square(*p);
    r += u.x;
  }
  return r;
}

double array(double x) {
  double r = 0;
  for (int i = 0; i < 3; ++i) {
    NonAssignable u[2] = {{x + i}, {2 * x}};
    r += u[0].x * u[1].x;
  }
  return r;
}

// CHECK-LABEL: void array_grad(double x, double *_d_x) {
// CHECK: clad::tape<NonAssignable[2]> [[A:_t[0-9]+]] = {};
// CHECK: clad::push([[A]], u) , ::new (static_cast<void *>(clad_addressof(u))) NonAssignable [2UL]{{.}}{x + i}, {2 * x}};
// CHECK: clad::reconstruct(u, clad::back([[A]]));
// CHECK: }

struct Number {
  double x;
  Number() = default;
  explicit Number(double value) : x(value) {}
  Number& operator=(const Number&) = delete;
};

double constructor(double x) {
  double r = 0;
  for (int i = 0; i < 3; ++i) {
    Number u(x + i);
    r += u.x * u.x;
  }
  return r;
}

// A nested scope that is not a loop moves its locals too.
double branch(double x) {
  double r = 0;
  if (x > 0) {
    NonAssignable u{x};
    r = u.x * u.x;
  }
  return r;
}

double nested(double x) {
  double r = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      NonAssignable u{x * (i + j)};
      r += u.x * u.x;
    }
  return r;
}

int main() {
  double dx = 0;
  clad::gradient(used).execute(4., &dx);
  std::printf("%.10g\n", dx); // CHECK-EXEC: 2

  dx = 0;
  clad::gradient(declarators).execute(4., &dx);
  std::printf("%.10g\n", dx); // CHECK-EXEC-NEXT: 6

  dx = 0;
  clad::gradient(distinct).execute(4., &dx);
  std::printf("%.10g\n", dx); // CHECK-EXEC-NEXT: 30

  dx = 0;
  clad::gradient(inactive).execute(4., &dx);
  std::printf("%.10g\n", dx); // CHECK-EXEC-NEXT: 3

  dx = 0;
  clad::gradient(by_reference).execute(4., &dx);
  std::printf("%.10g\n", dx); // CHECK-EXEC-NEXT: 30

  dx = 0;
  clad::gradient(array).execute(2., &dx);
  std::printf("%.10g\n", dx); // CHECK-EXEC-NEXT: 30

  dx = 0;
  clad::gradient(constructor).execute(4., &dx);
  std::printf("%.10g\n", dx); // CHECK-EXEC-NEXT: 30

  dx = 0;
  clad::gradient(branch).execute(4., &dx);
  std::printf("%.10g\n", dx); // CHECK-EXEC-NEXT: 8

  dx = 0;
  clad::gradient(nested).execute(3., &dx);
  std::printf("%.10g\n", dx); // CHECK-EXEC-NEXT: 36
}
