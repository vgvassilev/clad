// Debug info for code clad wrote.
//
// A generated node has no location of its own, so clad gives it one from a
// buffer of its own -- and that has to survive all the way into DWARF, or a
// debugger stopping inside a derivative has nothing to show. Before this,
// generated instructions were attributed to whatever the user happened to
// have on line one of their file.
//
// REQUIRES: llvm-dwarfdump
// RUN: %cladclang -g -gdwarf-5 -O0 -c %s -I%S/../../include -o %t.o
// RUN: %llvm-dwarfdump --debug-line %t.o | %filecheck %s
//
// Two generated statements do not report one position. Counted rather than
// pinned to particular lines: which line a statement lands on depends on how
// much code clad wrote before it, and asserting that would test the
// generator's output instead of the property. Every row the buffer owns,
// deduplicated by line -- one would mean every statement shares a position,
// which is the state this replaces.
//
// A handful rather than thousands: a slot is presented as the line of
// generated code it stands for, so this counts lines of a derivative, not
// slots handed out.
// RUN: %llvm-dwarfdump --debug-line %t.o | awk '\
// RUN:   BEGIN { want = -1 } \
// RUN:   /file_names\[/ { idx++ } \
// RUN:   /name: "<clad generated code>"/ { want = idx - 1 } \
// RUN:   /^0x[0-9a-f]+ / && $4 == want { line[$2] = 1 } \
// RUN:   END { n = 0; for (l in line) n++; print "distinct-lines", n }' \
// RUN:   | %filecheck --check-prefix=DISTINCT %s
// DISTINCT: distinct-lines {{([2-9]|[1-9][0-9]+)}}

#include "clad/Differentiator/Differentiator.h"

// The generated buffer is a file as far as the line table is concerned, and it
// is not the user's.
// CHECK-DAG: name: "<clad generated code>"

// A call in the body: the pullback of a call is where clad generates the most
// code that is not a rewrite of something the user wrote.
double helper(double x) { return x * x; }

double f(double x) {
  double t = helper(x);
  t = t * x;
  return t;
}

// A derivative with more generated nodes than one buffer holds continues in
// another, since a buffer cannot grow once the SourceManager holds it. Each
// restarts line numbering at 1, and the line table keys a file by its name --
// so buffers sharing a name fold into one entry and their lines collide,
// putting two different statements at the same file and line. Naming them
// apart is what keeps a position unambiguous.
// CHECK-DAG: name: "<clad generated code #2>"

// Enough statements to spill past one buffer, without a file of that length.
// Each one generates nodes in both sweeps, so this needs far fewer statements
// than a buffer holds slots -- these fill seven of them. The margin is
// deliberate: at only two, a target that generated somewhat fewer nodes per
// statement would fit in one buffer and fail this for the wrong reason.
#define CLAD_S1 t = t * 1.0000000001 + x * 0.5;
#define CLAD_S8 CLAD_S1 CLAD_S1 CLAD_S1 CLAD_S1 CLAD_S1 CLAD_S1 CLAD_S1 CLAD_S1
#define CLAD_S64 CLAD_S8 CLAD_S8 CLAD_S8 CLAD_S8 CLAD_S8 CLAD_S8 CLAD_S8 CLAD_S8
#define CLAD_S512                                                              \
  CLAD_S64 CLAD_S64 CLAD_S64 CLAD_S64 CLAD_S64 CLAD_S64 CLAD_S64 CLAD_S64
#define CLAD_S2048 CLAD_S512 CLAD_S512 CLAD_S512 CLAD_S512

double big(double x, double y) {
  double t = x * y;
  CLAD_S2048
  return t;
}

int main() {
  auto g = clad::gradient(f);
  double d = 0;
  g.execute(2, &d);

  auto gbig = clad::gradient(big);
  double dx = 0, dy = 0;
  gbig.execute(1.0, 1.0, &dx, &dy);
  return 0;
}
