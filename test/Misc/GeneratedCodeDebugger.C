// A debugger stopping inside code clad wrote.
//
// Locations in DWARF are only half of it. The other half is that the file
// those locations name has to exist and hold the code, or the debugger stops
// somewhere it cannot show anything. This runs the chain end to end: compile
// with debug information and somewhere to put the generated code, break on
// the derivative by name, and check where lldb says it stopped and what it
// prints there.
//
// By name rather than by line: which line a statement lands on depends on how
// much code clad generated ahead of it, and pinning that would make this a
// test of the generator's output instead of the debugging chain.
//
// REQUIRES: lldb
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %cladclang -g -O0 -Xclang -plugin-arg-clad \
// RUN:   -Xclang -fgenerated-source-dir=%t.dir -c %s -I%S/../../include \
// RUN:   -o %t.o
// RUN: %cladclang %t.o -o %t.out
// RUN: %lldb -b -o "breakpoint set --name f_grad" -o run -o "frame info" \
// RUN:   -o quit %t.out 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

// A call in the body, so the derivative is more than a rewrite of what the
// user wrote: the pullback of a call is where clad generates the most code of
// its own, and generated code is what this is about.
double helper(double x) { return x * x; }

double f(double x) {
  double t = helper(x);
  t = t * x;
  return t;
}

int main() {
  auto g = clad::gradient(f);
  double d = 0;
  g.execute(2, &d);
  return 0;
}

// The breakpoint resolves, so the name reached the line table.
// CHECK: Breakpoint 1: {{[0-9]+}} location

// The frame is in the file clad wrote, not in the one the user wrote.
// CHECK: f_grad{{.*}} at {{.*}}GeneratedCodeDebugger.C.clad.cpp:{{[0-9]+}}

// And lldb reads that file back off disk and shows the line it stopped on --
// the half gdb needs, since it cannot read code out of a binary at all.
// CHECK: {{->}} {{[0-9]+}}{{.*}}f_grad

// Without anywhere to write the code, the same run has a file name that
// answers to nothing, and shows no source at all. This is what the flag is
// for, and what the rest of this test would otherwise be passing without.
// RUN: %cladclang -g -O0 -c %s -I%S/../../include -o %t.nofile.o
// RUN: %cladclang %t.nofile.o -o %t.nofile.out
// RUN: %lldb -b -o "breakpoint set --name f_grad" -o run -o "frame info" \
// RUN:   -o quit %t.nofile.out 2>&1 | %filecheck --check-prefix=CHECK-NONE %s
// CHECK-NONE: f_grad{{.*}} at <clad generated code>:{{[0-9]+}}
// CHECK-NONE-NOT: {{->}} {{[0-9]+}}
