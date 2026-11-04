// CLAD_NO_NUM_DIFF used to turn the numerical-diff fallback off. It no longer
// does, and ignoring it would turn the fallback back on without a word: with
// the fallback on, it is an error. With the fallback off it asks for nothing
// more, so a build line can pass both the macro and the flag.
// RUN: %cladnumdiffclang -fsyntax-only -Xclang -verify %s -I%S/../../include
// RUN: %cladclang -fsyntax-only -Xclang -verify -DFALLBACK_OFF %s \
// RUN:   -I%S/../../include

#ifdef FALLBACK_OFF
// expected-no-diagnostics
#define CLAD_NO_NUM_DIFF
#else
#define CLAD_NO_NUM_DIFF // expected-error {{CLAD_NO_NUM_DIFF no longer turns the numerical-diff fallback off; pass -fno-numerical-diff-fallback to clad instead (-Xclang -plugin-arg-clad -Xclang -fno-numerical-diff-fallback)}}
#endif

#include "clad/Differentiator/Differentiator.h"

double f(double x) { return x * x; }

void request() { clad::differentiate(f, "x"); }
