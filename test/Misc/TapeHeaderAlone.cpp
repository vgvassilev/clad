// RUN: %cladclang %s -I%S/../../include -oTapeHeaderAlone.out
// RUN: ./TapeHeaderAlone.out | %filecheck_exec %s

// Tape.h declares clad::tape once, with its defaults, so it works on its own.
// A second declaration in Differentiator.h is one GCC rejects.

#include "clad/Differentiator/Tape.h"

#include <cstdio>

int main() {
  clad::tape<double> t;
  t.emplace_back(1.5);
  printf("%.1f\n", t.back()); // CHECK-EXEC: 1.5
}
