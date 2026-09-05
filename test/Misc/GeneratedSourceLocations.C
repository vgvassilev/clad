// Clad-generated code has no source location of its own, and the placeholder
// every visitor reaches for is the start of the main file -- so a diagnostic
// about generated code puts its caret on the user's first line.
//
// Rendering the derivative into a buffer registered with the SourceManager
// gives each generated statement a real location instead. This checks the
// round-trip: the line and column reported for a statement is the one that
// statement's text actually occupies in the rendering.
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-generated-source \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

double f(double x) {
  double t = x * x;
  t = t * t;
  return t;
}

// The forward sweep, in the order the rendering lays it out. Each node reports
// the span it occupies, not just where it starts: a diagnostic underlines the
// whole expression, which is what makes a multi-node one -- a loop condition,
// an increment -- read as one thing.
// CHECK: generated-source: f_grad
// CHECK: {{[0-9]+}}:5-{{[0-9]+}}: double _d_t = 0.;
// CHECK-NEXT: {{[0-9]+}}:5-{{[0-9]+}}: double t = x * x;
// CHECK-NEXT: {{[0-9]+}}:5-18: double _t0 = t;

// The nested expressions of `t = t * t` sit further in on the same line, and
// each spans exactly its own text: the assignment reaches column 13, its
// right-hand side starts at 9, and the operand at 13 is one character wide.
// CHECK-NEXT: [[LINE:[0-9]+]]:5-13: t = t * t;
// CHECK-NEXT: [[LINE]]:9-13: t * t;
// CHECK-NEXT: [[LINE]]:13-13: t;

// The reverse sweep restores what the forward one saved.
// CHECK: {{[0-9]+}}:9-{{[0-9]+}}: t = _t0;

// One buffer holds every derivative in the unit, so only the first of them
// starts where the buffer does. A second one is where a position into the
// buffer and a position into the text stop agreeing, and reporting the text
// by the wrong one of the two prints nothing at all.
// CHECK: generated-source: g_grad
// CHECK: {{[0-9]+}}:5-{{[0-9]+}}: double _d_u = 0.;
// CHECK-NEXT: {{[0-9]+}}:5-{{[0-9]+}}: double u = y * y;

// A second function, so that a second derivative is printed into the same
// buffer after the first.
double g(double y) {
  double u = y * y;
  u = u * u;
  return u;
}

int main() {
  auto gf = clad::gradient(f);
  auto gg = clad::gradient(g);
  double d = 0;
  gf.execute(2, &d);
  gg.execute(2, &d);
  return 0;
}
