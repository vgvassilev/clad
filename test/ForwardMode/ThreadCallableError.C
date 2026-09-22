// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <functional>
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

void thread_noop() {}

// Outside the differentiated fn so we only hit the thread diagnostic.
std::function<void()> g_std_function = thread_noop;

double f_std_function(double x) {
  std::thread t(g_std_function); // expected-error {{failed to resolve callable of type 'std::function}}
  t.join();
  return x * x;
}

double f_lambda(double x) {
  auto worker = []() {};
  // expected-warning@-1 0-1 {{statement kind 'LambdaExpr' is not supported}}
  // expected-error@* 0-2 {{declaration of variable '_d_worker' with deduced type 'auto' requires an initializer}}
  // expected-note@-3 0-20 {{unnamed type used in template argument was declared here}}
  std::thread t(worker); // expected-error {{forward-mode differentiation of std::thread with a lambda callable is not supported yet}}
  t.join();
  return x * x;
}

double f_detach(double x) {
  std::thread t(thread_noop);
  t.detach(); // expected-error {{detach is not supported in forward-mode AD of std::thread}}
  return x * x;
}

int main() {
  clad::differentiate(f_unresolved, "x");
  clad::differentiate(f_std_function, "x");
  clad::differentiate(f_lambda, "x");
  clad::differentiate(f_detach, "x");
  return 0;
}
