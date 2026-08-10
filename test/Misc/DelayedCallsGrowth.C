// RUN: %cladclang -I%S/../../include %s -oDelayedCallsGrowth.out
// RUN: ./DelayedCallsGrowth.out | %filecheck_exec %s

// Planning a request whose target is a constructor (or a conversion operator)
// makes clad instantiate a clad::Tag<T> specialization -- see
// utils::GetDerivativeType -- and Sema reports that new tag back through the
// plugin's HandleTagDeclDefinition, which appends to m_DelayedCalls. The
// planning loop in CladPlugin::HandleTranslationUnit walks that very deque, so
// differentiating a constructor grows the container mid-iteration.
//
// Inserting into a deque invalidates every iterator to it, though not
// references to its elements ([deque.modifiers]/1,
// https://eel.is/c++draft/deque.modifiers#1). The loop used to be a range-for,
// whose end() is computed once up front, so from the first such append onwards
// it walked invalidated iterators and eventually dereferenced a DeclGroupRef
// read out of bounds.
//
// Note this test pins the code path; it cannot reliably reproduce the crash.
// Whether the stale iterator faults depends on the deque reallocating its block
// map at that moment, which is a function of how many delayed calls precede the
// append and of the standard library's map slack -- not of anything this
// source can control.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct S {
  double x;
  // Differentiating through this constructor is what triggers the
  // planning-time clad::Tag<S> instantiation.
  S(double v) : x(v) {}
};

double f(double u) {
  S s(u);
  return s.x * s.x;
}

int main() {
  auto g = clad::gradient(f);
  double dx = 0;
  g.execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  return 0;
}
