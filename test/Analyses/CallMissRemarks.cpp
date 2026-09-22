// What a data-flow analysis gives up on, reported the same way a recogniser
// reports it.
//
// The to-be-recorded analysis can skip saving an argument a callee never
// reads, but only where it has read that callee. Where it has not, every
// argument of the call is saved, and the only person who can change that is
// the one who wrote the call.
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=tbr \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 | %filecheck %s
//
// Asking about a different analysis says nothing about calls: each report is
// gated by the analysis that filed it.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=loop \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-LOOP %s

#include "clad/Differentiator/Differentiator.h"

// No body here, so nothing says which of its arguments it reads.
double mix(double x, double y);

namespace clad {
namespace custom_derivatives {
void mix_pullback(double x, double y, double d_z, double* d_x, double* d_y) {
  *d_x += d_z;
  *d_y += d_z;
}
} // namespace custom_derivatives
} // namespace clad

// This one has a body, so clad reads it and says nothing.
double square(double x) { return x * x; }

// Nothing here has to be kept for the reverse sweep, so the only thing the
// report has to say is what it could not read.
double f(double x) { return mix(x, 2.) + square(x); }

// The caret is on the call, which is where the cost lands.
// CHECK: CallMissRemarks.cpp:[[# @LINE - 3]]:29: remark: clad saves every argument of this call for the reverse sweep
// CHECK-NEXT: double f(double x) { return mix(x, 2.) + square(x); }
// CHECK: note: this function has no body in this file
// CHECK: note: to avoid this, make it a call clad can look inside (CLAD1004)

// Nothing is said about square: clad read it, so this is not a miss, and a
// reader sent to look at it would find nothing to fix.
// CHECK-NOT: CallMissRemarks.cpp:[[# @LINE - 10]]:42

// CHECK-LOOP-NOT: remark: clad saves every argument

int main() { auto g = clad::gradient(f); }
