// RUN: %cladclang %s -I%S/../../include -oStructuredBindings.out 2>&1 | %filecheck %s
// RUN: ./StructuredBindings.out | %filecheck_exec %s
// RUN: not %cladclang %s -I%S/../../include -fsyntax-only -DNOTYET 2>&1 \
// RUN:   | FileCheck --check-prefix=NOTYET %s

// A structured binding names parts of a variable that has no name of its own.
// The derivative used to lose both and came out zero. What is not supported
// yet is an error.

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"
#include <array>
#include <cstdio>
#include <utility>

struct Two {
  double a, b;
};

double byValue(double x) {
  auto [a, b] = Two{x, 2 * x};
  return a * b;
}

// CHECK: double byValue_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Two _d__bind = Two({_d_x, 2 * _d_x});
// CHECK-NEXT:     Two _bind0 = Two({x, 2 * x});

// Through a reference, a write to a binding is a write to the object.
double byReference(double x) {
  Two s{x, 1};
  auto& [a, b] = s;
  a *= 2;
  return s.a * x;
}

// A tuple-like type introduces a variable for each binding, initialized with
// get<i> ([dcl.struct.bind]), and differentiated as any other.
double tupleByValue(double x) {
  auto [a, b] = std::pair<double, double>{x, 2 * x};
  return a * b;
}

// CHECK: double tupleByValue_darg0(double x) {
// CHECK: std::tuple_element<0{{(UL)?}}, std::pair<double, double> >::type &&_d_a = {{.*}};
// CHECK: std::tuple_element<0{{(UL)?}}, std::pair<double, double> >::type &&a = {{.*}};

double tupleByReference(double x) {
  std::array<double, 2> v{x, 3 * x};
  auto& [p, q] = v;
  p *= 2;
  return v[0] * q;
}

// The element of a const object is read, and its adjoint written, through
// the object's adjoint.
double tupleConst(double x) {
  auto const [a, b] = std::pair<double, double>{x, 2 * x};
  return a * b;
}

#ifdef NOTYET
// The elements of an array copy would be read as zero.
double arrayCopy(double x) {
  double v[2] = {x, 0.7};
  auto [p, q] = v;
  return p * x + q;
}
// NOTYET: error: differentiating a structured binding of an array copy is not supported yet
// NOTYET: error: differentiating a structured binding of an array copy is not supported yet
// NOTYET-NOT: error:
auto d_arrayCopy = clad::differentiate(arrayCopy, "x");
auto g_arrayCopy = clad::gradient(arrayCopy);
#endif

int main() {
  double x = 3, d;
  auto f1 = clad::differentiate(byValue, "x");
  auto g1 = clad::gradient(byValue);
  d = 0;
  g1.execute(x, &d);
  // 2x^2: 4x
  std::printf("%g %g\n", f1.execute(x), d); // CHECK-EXEC: 12 12
  auto f2 = clad::differentiate(byReference, "x");
  auto g2 = clad::gradient(byReference);
  d = 0;
  g2.execute(x, &d);
  // 2x^2: 4x
  std::printf("%g %g\n", f2.execute(x), d); // CHECK-EXEC: 12 12
  auto f3 = clad::differentiate(tupleByValue, "x");
  auto g3 = clad::gradient(tupleByValue);
  d = 0;
  g3.execute(x, &d);
  // 2x^2: 4x
  std::printf("%g %g\n", f3.execute(x), d); // CHECK-EXEC: 12 12
  auto f4 = clad::differentiate(tupleByReference, "x");
  auto g4 = clad::gradient(tupleByReference);
  d = 0;
  g4.execute(x, &d);
  // 6x^2: 12x
  std::printf("%g %g\n", f4.execute(x), d); // CHECK-EXEC: 36 36
  auto f5 = clad::differentiate(tupleConst, "x");
  auto g5 = clad::gradient(tupleConst);
  d = 0;
  g5.execute(x, &d);
  // 2x^2: 4x
  std::printf("%g %g\n", f5.execute(x), d); // CHECK-EXEC: 12 12
}
