// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1
// CHECK-NOT: {{.*warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double func1(double x, double y) {
  return x * x + y * y;
}

int main () {
  auto func1_dx = clad::differentiate(func1, "x");

  func1_dx.execute(5);
  // expected-error@clad/Differentiator/Differentiator.h:* {{too few arguments to function call, expected 2, have 1}}
  // expected-note@clad/Differentiator/Differentiator.h:* {{in instantiation of function template specialization 'clad::execute_with_default_args<false, double, double (*)(double, double), int, double, true>' requested here}}
#if __clang_major__ < 16
  // expected-note@clad/Differentiator/Differentiator.h:* {{in instantiation of function template specialization 'clad::CladFunction<double (*)(double, double), clad::NoObject, false>::execute_helper<double (*)(double, double), int>' requested here}}
  // expected-note@NotEnoughArgError.C:13 {{in instantiation of function template specialization 'clad::CladFunction<double (*)(double, double), clad::NoObject, false>::execute<int, double (*)(double, double)>' requested here}}
#else
  // expected-note@clad/Differentiator/Differentiator.h:* {{in instantiation of function template specialization 'clad::CladFunction<double (*)(double, double), clad::NoObject>::execute_helper<double (*)(double, double), int>' requested here}}
  // expected-note@NotEnoughArgError.C:13 {{in instantiation of function template specialization 'clad::CladFunction<double (*)(double, double), clad::NoObject>::execute<int, double (*)(double, double)>' requested here}}
#endif
}
