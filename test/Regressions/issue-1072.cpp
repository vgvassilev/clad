// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -I%S/../../include %s -fsyntax-only 2>&1 | %filecheck %s
// RUN: %cladclang -I%S/../../include %s -fsyntax-only 2>&1 | %filecheck %s

// clad declares a temporary for the index argument a pullback writes back to.
// Printed as a bare `size_type` it is not a name anything can look up, so the
// declaration has to say which container the type belongs to -- or use a name
// that stands on its own, which is what `size_t` does. Which of the two a
// platform prints varies, so the check below takes either and refuses only
// the unqualified form. A const vector reaches the pullback, which is where
// such a temporary is declared.

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <vector>

double fn(const std::vector<double>& v) { return v.at(0); }

// CHECK: void fn_grad(const std::vector<double> &v, std::vector<double> *_d_v) {
// CHECK: {{(size_t|std::vector<.*>::size_type)}} _r0 = 0{{U|UL|ULL}};
// CHECK: at_pullback(&v, {{.*}}&_r0);

int main() {
  auto grad = clad::gradient(fn);
}
