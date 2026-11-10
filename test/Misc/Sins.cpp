// RUN: %cladclang %s -I%S/../../include -oSins.out
// RUN: ./Sins.out | %filecheck_exec %s

// CLAD_ACCESS reads a private member through the pointer CLAD_ALLOW_ACCESS
// obtains in an explicit instantiation, where access checking does not apply
// ([temp.spec.general]).

#include "clad/Differentiator/Sins.h"

#include <cstdio>

class Box {
  double v[2] = {1.5, 2.5};
  double s = 4;
};

// A type that does not read right before `Box::*`, an array, is named by an
// alias.
using Pair = double[2];
CLAD_ALLOW_ACCESS(Box, v, Pair);
class Other {
  int s = 7;
};
// Members of one name opened on one line, as two headers may.
// clang-format off
CLAD_ALLOW_ACCESS(Box, s, double); CLAD_ALLOW_ACCESS(Other, s, int);
// clang-format on

// A const object: Only_ is specialized for the class, not for `const Box`.
double first(const Box& b) { return CLAD_ACCESS(b, v)[0]; }

template <class T> double scaled(T& b) { return 2 * CLAD_ACCESS(b, s); }

int main() {
  Box b;
  // `[]` after CLAD_ACCESS applies to the member, not to the pointer to it.
  CLAD_ACCESS(b, v)[1] = 3.5;
  Other o;
  std::printf("%g %g %g %d\n", first(b), CLAD_ACCESS(b, v)[1], scaled(b),
              CLAD_ACCESS(o, s));
  // CHECK-EXEC: 1.5 3.5 8 7
}
