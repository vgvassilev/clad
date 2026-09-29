// clad::opts bits are baked into every translation unit that names one, so a
// number that is reused or does not fit has to stop the build. Both checks sit
// in a public header and fire wherever it is read; these confirm they fire,
// against a stand-in table found ahead of the real one on the include path --
// which is why they drive the compiler directly: %cladclang names the
// directory the real table is rendered into, and would find that first.

// RUN: not clang++ -std=c++17 -fsyntax-only -I%S/Inputs/opts-collide \
// RUN:   -I%S/../../include %s 2>&1 | FileCheck --check-prefix=COLLIDE %s
// COLLIDE: two clad::opts share a bit

// A position is counted from ORDER_BITS, so one below zero would put the pair
// where the derivative order is read from -- where it collides with no option
// and quietly changes what a request asks for.
// RUN: not clang++ -std=c++17 -fsyntax-only -I%S/Inputs/opts-negative \
// RUN:   -I%S/../../include %s 2>&1 | FileCheck --check-prefix=NEGATIVE %s
// NEGATIVE: the position for loop is below the order

// RUN: not clang++ -std=c++17 -fsyntax-only -I%S/Inputs/opts-overflow \
// RUN:   -I%S/../../include %s 2>&1 | FileCheck --check-prefix=OVERFLOW %s
// OVERFLOW: clad::opts has run out of bits: the pair for loop does not fit

#include "clad/Differentiator/CladConfig.h"
