// RUN: %cladclang -O1 -Xclang -fdebug-pass-manager %s -I%S/../../include \
// RUN:   -oPipelineOptIn.out 2>&1 | %filecheck %s
// RUN: ./PipelineOptIn.out | %filecheck_exec %s
// REQUIRES: Enzyme

// Enzyme's passes belong only in a translation unit that asked for them.
// clad has finished deriving by the time clang builds the backend pipeline,
// so the plugin registers Enzyme only when something here named the backend.
// Compiling everything else through a third party pipeline is not clad's to
// decide on the user's behalf.
//
// This file does ask, so the passes are expected. Nothing.C is the other half.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double f(double x, double y) { return x * x * y; }

// What is registered is Enzyme's canonical pipeline, not a bare pass spliced
// in at pipeline start: its own pre-simplification runs with it. Order-free,
// since where each lands in the pipeline is Enzyme's business, not clad's.
// CHECK-DAG: Running pass: PreserveNVVM
// CHECK-DAG: Running pass: EnzymeNewPM

int main() {
  auto g = clad::gradient<clad::opts::use_enzyme>(f);
  double dx = 0, dy = 0;
  g.execute(3, 4, &dx, &dy);
  printf("{%.2f, %.2f}\n", dx, dy);
  // CHECK-EXEC: {24.00, 9.00}
}
