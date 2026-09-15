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

double f_lambda(double x) {
  auto worker = []() {};
  // expected-warning@-1 0-1 {{statement kind 'LambdaExpr' is not supported}}
  // Older clangs may leave an uninitialized deduced adjoint for the lambda:
  // expected-error@* 0-2 {{declaration of variable '_d_worker' with deduced type 'auto' requires an initializer}}
  // Newer clangs may note the lambda type when used in generated templates:
  // expected-note@-4 0-20 {{unnamed type used in template argument was declared here}}
  std::thread t(worker); // expected-error {{reverse-mode differentiation of std::thread with a lambda callable is not supported yet}}
  t.join();
  return x * x;
}

void thread_noop() {}

double f_detach(double x) {
  std::thread t(thread_noop);
  t.detach(); // expected-error {{detach is not supported in reverse-mode AD of std::thread}}
  return x * x;
}

int main() {
  clad::gradient(f_unresolved);
  clad::gradient(f_lambda);
  clad::gradient(f_detach);
  return 0;
}
