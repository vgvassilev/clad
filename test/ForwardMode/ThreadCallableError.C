// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <thread>

// Multiple call operators: resolveThreadCallable cannot pick one.
struct OverloadedCallable {
  void operator()() const {}
  void operator()(int) const {}
};

double f_unresolved(double x) {
  std::thread t(OverloadedCallable{}); // expected-error {{failed to resolve callable of type 'OverloadedCallable' passed to std::thread}}
  t.join();
  return x * x;
}

int main() {
  clad::differentiate(f_unresolved, "x");
  return 0;
}
