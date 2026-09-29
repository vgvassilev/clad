// An analysis is asked for by the name it goes by everywhere else, and by the
// name the first of these was given. clad::opts::enable_va says nothing about
// which analysis it is; the short spellings stay because a position reaches the
// mangled name of every request that carries one, so they are the same option
// under two names.

// RUN: %cladclang %s -I%S/../../include -oOptsSpellings.out 2>&1 | %filecheck %s
// RUN: ./OptsSpellings.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// The two spellings of one option are one value, not two. Checked for every
// analysis in the table rather than for the one that prompted this, so a new
// analysis is covered by existing.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, FirstBit, Desc)               \
  static_assert(clad::opts::enable_##Name##_analysis ==                        \
                    clad::opts::enable_##Legacy,                               \
                "the two spellings of enable_" #Legacy " differ");             \
  static_assert(clad::opts::disable_##Name##_analysis ==                       \
                    clad::opts::disable_##Legacy,                              \
                "the two spellings of disable_" #Legacy " differ");
#include "clad/Differentiator/Analyses.def"

double sum(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += x[i] * x[i];
  return s;
}

int main() {
  double x[3] = {1, 2, 3};
  double dx[3] = {0, 0, 0};
  int dn = 0;
  // The long spelling reaches the request, not just the enum: asked the way
  // the short one is asked in LoopAnalysisRequestSwitch.cpp.
  auto g = clad::gradient<clad::opts::disable_loop_analysis>(sum);
  g.execute(x, 3, dx, &dn);
  printf("%.1f %.1f %.1f\n", dx[0], dx[1], dx[2]);
  // CHECK-EXEC: 2.0 4.0 6.0
}
