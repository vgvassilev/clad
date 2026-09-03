// A derivative is printed into the buffer its own nodes point into.
//
// Clad hands out locations from a buffer as it builds a derivative and starts
// a new buffer when that one runs out. The printed code has to go into the
// buffer that derivative's nodes came from, not into whichever buffer happens
// to be current when the printing is done: a note on a node can only name a
// line of the file that node is in.
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=tbr \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

// Differentiated first, so its nodes are in the first buffer.
double first(double x) {
  double t = x * x;
  t = t * t;
  return t;
}

// Enough nodes to run past the first buffer and into later ones. Linear, so
// the reverse sweep keeps nothing here and this reports nothing of its own.
#define B1 t = t + x;
#define B8 B1 B1 B1 B1 B1 B1 B1 B1
#define B64 B8 B8 B8 B8 B8 B8 B8 B8
#define B512 B64 B64 B64 B64 B64 B64 B64 B64
#define B2048 B512 B512 B512 B512
double bulk(double x) {
  double t = x;
  B2048
  return t;
}

// Differentiated last, so its nodes are in a later buffer.
double last(double x) {
  double t = x * x;
  t = t * t;
  return t;
}

int main() {
  double d = 0;
  auto gf = clad::gradient(first);
  auto gb = clad::gradient(bulk);
  auto gl = clad::gradient(last);
  gf.execute(2, &d);
  gb.execute(2, &d);
  gl.execute(2, &d);
  return 0;
}

// The first derivative reports the first buffer, which has no number.
// CHECK: <clad generated code>:{{[0-9]+}}:{{[0-9]+}}: remark: clad keeps this value for the reverse sweep
// CHECK: note: to-be-recorded analysis could not show it unused
// CHECK: {{.*}}GeneratedCodeChunks.C:[[@LINE-12]]:13: note: in the derivative of 'first' requested here
// CHECK: {{.*}}GeneratedCodeChunks.C:[[@LINE-39]]:3: note: the value kept is the one this expression had

// And the last one a later buffer. That is also what says there was more than
// one buffer to get wrong: with a single buffer this would read as the first.
// CHECK: <clad generated code #{{[0-9]+}}>:{{[0-9]+}}:{{[0-9]+}}: remark: clad keeps this value for the reverse sweep
// CHECK: note: to-be-recorded analysis could not show it unused
// CHECK: {{.*}}GeneratedCodeChunks.C:[[@LINE-17]]:13: note: in the derivative of 'last' requested here
// CHECK: {{.*}}GeneratedCodeChunks.C:[[@LINE-26]]:3: note: the value kept is the one this expression had
