// A remark about generated code, pointed at the generated code.
//
// Clad's analyses decide what the derivative has to keep, and when one cannot
// prove a value dead the user pays for it with no way to find out where. This
// is the -Rpass-missed analogue: clang's own remark machinery runs in the
// backend over LLVM IR and never sees an AST-level plugin, so clad carries its
// own switch and its own locations.
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=tbr \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:   | %filecheck --check-prefixes=CHECK,CHECK-NOKEEP %s
//
// The caret lands inside the generated code, on the statement itself -- the
// whole point, since none of this text exists in a file the user wrote. One
// buffer holds every derivative in the unit, so which one this is comes from
// the note below rather than from the buffer's name.
// CHECK: <clad generated code>:{{[0-9]+}}:{{[0-9]+}}: remark: clad keeps this value for the reverse sweep
// CHECK: double _t0 = t;
// The whole statement is underlined, not just its first character: a range
// reads as one thing where a caret reads as a position.
// CHECK-NEXT: ^~~~~~~~~~~~~~
// CHECK: note: to-be-recorded analysis could not show it unused
//
// Which derivative this is, said in the diagnostic rather than left to the
// include stack: a caret on the differentiation, and the name of what is
// being differentiated.
// CHECK: {{.*}}AnalysisRemarks.C:[[@LINE+110]]:12: note: in the derivative of 'f' requested here
// CHECK-NEXT: clad::gradient(f)
//
// And the half the user can act on -- their own code, with the expression
// whose value is being kept.
// CHECK: {{.*}}AnalysisRemarks.C:[[@LINE+86]]:3: note: the value kept is the one this expression had
// CHECK-NEXT: t = t * t;
//
// A value the reverse sweep needs once per iteration cannot go into a
// variable of its own, so clad pushes it onto a tape. The push is what
// costs, and it is what the remark has to point at -- not the tape, which
// is only the container.
// CHECK: <clad generated code>:{{[0-9]+}}:{{[0-9]+}}: remark: clad keeps this value for the reverse sweep
// CHECK: clad::push(
// Directly under it the underline, and nothing else. Clang reads the line to
// show out of the buffer, at the number the remark gives, so a derivative
// printed somewhere other than where it says it is shows up right here: the
// next line of the buffer instead of the underline.
// CHECK-NEXT: {{\^~+$}}
// CHECK: note: to-be-recorded analysis could not show it unused
// CHECK: note: in the derivative of 'loopy' requested here
// CHECK: {{.*}}AnalysisRemarks.C:[[@LINE+80]]:5: note: the value kept is the one this expression had
// CHECK-NEXT: t = t * t;
//
// Not every derivative has a differentiation to name. The second derivative a
// hessian needs is one clad asks itself for, at no line the user wrote, so the
// note is left out rather than aimed at something arbitrary -- and the run's
// exhaustive note matching is what holds it to that.
//
// Nor does every kept value have an expression the user wrote. The first of
// these is clad's own derivative variable, so there is nothing of theirs to
// point at and the third note is left out too.
// CHECK: <clad generated code>:{{[0-9]+}}:{{[0-9]+}}: remark: clad keeps this value for the reverse sweep
// CHECK: note: to-be-recorded analysis could not show it unused
//
// The second is the user's own `t`, so that one is named, and by where they
// wrote it rather than by where clad rebuilt it.
// CHECK: <clad generated code>:{{[0-9]+}}:{{[0-9]+}}: remark: clad keeps this value for the reverse sweep
// CHECK: note: to-be-recorded analysis could not show it unused
// CHECK: {{.*}}AnalysisRemarks.C:[[@LINE+52]]:3: note: the value kept is the one this expression had
// CHECK-NEXT: t = t * t;
//
// With the analysis switched off the value is kept for a different reason, and
// the note says which. Claiming it "could not prove" something it never ran
// would be false.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=tbr \
// RUN:   -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=tbr \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-OFF %s
// CHECK-OFF: <clad generated code>:{{[0-9]+}}:{{[0-9]+}}: remark: clad keeps this value for the reverse sweep
// CHECK-OFF: note: to-be-recorded analysis is disabled
// Neither pointer back into the user's code depends on the analysis: which
// derivative this is, and the expression whose value is kept.
// CHECK-OFF: note: in the derivative of 'f' requested here
// CHECK-OFF: note: the value kept is the one this expression had
// Nor does the tape: what changes with the switch is the reason, not which
// values are reported or where they came from.
// CHECK-OFF: <clad generated code>:{{[0-9]+}}:{{[0-9]+}}: remark: clad keeps this value for the reverse sweep
// CHECK-OFF: note: to-be-recorded analysis is disabled
// CHECK-OFF: note: in the derivative of 'loopy' requested here
// CHECK-OFF: note: the value kept is the one this expression had
// Nor does the switch put back either note that was left out.
// CHECK-OFF: <clad generated code>:{{[0-9]+}}:{{[0-9]+}}: remark: clad keeps this value for the reverse sweep
// CHECK-OFF: note: to-be-recorded analysis is disabled
// CHECK-OFF: <clad generated code>:{{[0-9]+}}:{{[0-9]+}}: remark: clad keeps this value for the reverse sweep
// CHECK-OFF: note: to-be-recorded analysis is disabled
// CHECK-OFF: note: the value kept is the one this expression had
//
// A derivative with nothing to keep is not something to report: asking about
// it says nothing rather than saying there is nothing. By the note, since one
// buffer holds every derivative and its name says nothing about which.
// CHECK-NOKEEP-NOT: in the derivative of 'lin' requested here
//
// A name no analysis answers to is refused, and the refusal says which names
// there are. Plain FileCheck here: the run fails, so its output carries an
// error the shared %filecheck is set to reject.
// RUN: not %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=bogus \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-BADNAME %s
// CHECK-BADNAME: clad: Error: unknown analysis 'bogus'; known:{{.*}}tbr
//
// Nothing is rendered unless asked: no flag, no remark, and no buffer built.
// RUN: %cladclang -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-QUIET --allow-empty %s
// CHECK-QUIET-NOT: remark:
// CHECK-QUIET-NOT: clad generated code

#include "clad/Differentiator/Differentiator.h"

double f(double x) {
  double t = x * x;
  t = t * t;
  return t;
}

// A value the reverse sweep needs once per iteration cannot go into a variable
// of its own -- there is one of it per iteration -- so clad pushes it onto a
// tape instead.
double loopy(double x, int n) {
  double t = x;
  for (int i = 0; i < n; ++i)
    t = t * t;
  return t;
}

// Nothing here has to be kept: the reverse sweep reads no value this wrote.
double lin(double x) { return 2 * x; }

int main() {
  double d0 = 0;
  auto g = clad::gradient(f);
  auto gn = clad::gradient(lin);
  gn.execute(2, &d0);
  double d = 0;
  g.execute(2, &d);
  auto gl = clad::gradient(loopy, "x");
  gl.execute(2, 3, &d);
  // A hessian differentiates a derivative, and asks for that one itself.
  double h = 0;
  auto hf = clad::hessian(f);
  hf.execute(2, &h);
  return 0;
}
