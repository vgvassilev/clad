// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

double fn_test(double x) {
  return x * x;
}

struct FunctorTest {
  double operator()(double x) const { return x * x; }
};

#include <initializer_list>

double fn_init_list(std::initializer_list<double> l) { return 0.0; }

int main() {
  // 1. Enzyme option is not supported for pullback mode
  auto pb_enz = clad::pullback<clad::opts::use_enzyme>(fn_test); // expected-error {{enzyme is not supported for pullback mode}}

  // 2. Vector mode option is not supported for pullback mode
  auto pb_vec = clad::pullback<clad::opts::vector_mode>(fn_test); // expected-error {{reverse vector mode is not yet supported}}

  // 3. std::initializer_list is not supported for pullback mode
  auto pb_init = clad::pullback(fn_init_list); // expected-error {{pullback does not support functions with std::initializer_list parameters}}

  // 4. Lambda expressions are not supported for pullback mode (use named function objects)
  auto l_stateless = [](double x) { return x * x; };
  auto pb_l_stateless = clad::pullback(l_stateless); // expected-error {{pullback does not support lambda expressions; use named function objects instead}}
  auto pb_l_stateless_part = clad::pullback(l_stateless, "x"); // expected-error {{pullback does not support lambda expressions; use named function objects instead}}

  double a = 3.0;
  auto l_cap = [a](double x) { return a * x; };
  auto pb_l_cap = clad::pullback(l_cap); // expected-error {{pullback does not support lambda expressions; use named function objects instead}}
  auto pb_l_cap_part = clad::pullback(l_cap, "x"); // expected-error {{pullback does not support lambda expressions; use named function objects instead}}

  // 5. Invalid / malformed argument selection diagnostics
  auto pb_empty = clad::pullback(fn_test, ""); // expected-error {{no parameters were provided}}
  auto pb_invalid = clad::pullback(fn_test, "nonexistent"); // expected-error {{requested parameter name 'nonexistent' was not found among function parameters}}

  // 6. Subsequent valid differentiation succeeds cleanly in the same Sema instance
  auto pb_ok = clad::pullback(fn_test);
  double dx = 0.0;
  pb_ok.execute(3.0, 1.0, &dx);

  return 0;
}
