// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"

// The jacobian subscripts the output's adjoint through clad::matrix's
// operator[], and in this translation unit that subscript is what first
// instantiates its return type. With no source location for it, an
// assertion-enabled clang aborted. Nothing here may use clad::matrix itself,
// or the instantiation happens first in user code and the test observes
// nothing.
void output(double x, double out[]) { out[0] = x * x; }

void request() { clad::jacobian(output); }
