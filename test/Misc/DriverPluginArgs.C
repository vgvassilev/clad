// clang's own spelling for a plugin's options, which needs no -Xclang. The
// dash is doubled because clang hands the plugin whatever follows
// -fplugin-arg-clad- verbatim, and every clad option starts with one of its
// own; -fplugin-arg-clad-help would pass `help` and be rejected.
//
// The driver gained this in clang 14, which is the oldest clang clad
// supports, so every version reaching this test has it. -Xclang
// -plugin-arg-clad, which Args.C covers, is the other spelling.
//
// RUN: clang -fsyntax-only -fplugin=%cladlib -fplugin-arg-clad--help \
// RUN:   -I%S/../../include -I%clad_gen_incl %s 2>&1 | FileCheck %s
// The header is a line of its own, which it was not before the screen came
// from the table, and the option after it is the first the table holds.
// CHECK: Options specific to Clad (preceded by -plugin-arg-clad):
// CHECK-NEXT: -Rclad-analysis=<name>
// CHECK: -fdump-derived-fn -

// -fplugin= alone both loads clad and runs it; no -add-plugin is needed.
// RUN: clang -fplugin=%cladlib -fplugin-arg-clad--fdump-derived-fn \
// RUN:   -I%S/../../include -I%clad_gen_incl %s -o %t 2>&1 | FileCheck --check-prefix=CHECK-DUMP %s
// RUN: %t | %filecheck_exec %s
// CHECK-DUMP: double f_darg0(double x)

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double f(double x) { return x * x; }

int main() {
  auto g = clad::differentiate(f, "x");
  printf("%.1f\n", g.execute(3));
  // CHECK-EXEC: 6.0
}
