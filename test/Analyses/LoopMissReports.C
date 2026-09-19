// Every reason the loop analysis can give for not counting a `for` loop,
// except the two LoopAnalysisSwitch.C already shows with their carets: an
// increment that does not step by one, and a bound the body writes. No reason
// should reach a user without having been read once here.
//
// FileCheck rather than %filecheck: the wrapper rejects any line with the word
// "note:" in it, and a report is mostly notes. Syntax-only because what is
// under test is what clad says, not what it computes.
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=loop \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

// One loop per reason: the recogniser stops at the first thing that missed, so
// a loop can only ever report one.
double headers(const double* x, int n) {
  double s = 0;
  int outside = 0;
  int start = n;
  start = n - 1;

  // CHECK-DAG: note: the condition is not a comparison
  for (int i = 0; !(i >= n); i++)
    s += x[i];

  // CHECK-DAG: note: the comparison is not < or <=
  for (int i = 0; i != n; i++)
    s += x[i];

  // CHECK-DAG: note: the left side of the comparison is not a variable
  for (int i = 0; (i + 0) < n; i++)
    s += x[i];

  // A floating index makes the count depend on rounding.
  // CHECK-DAG: note: the index is not an integer
  for (double i = 0; i < n; i++)
    s += x[(int)i];

  // CHECK-DAG: note: the header gives the index no start value
  for (; outside < n; outside++)
    s += x[outside];

  // CHECK-DAG: note: the body can leave the loop early
  for (int i = 0; i < n; i++) {
    if (x[i] < 0)
      break;
    s += x[i];
  }

  // The increment is the only thing allowed to move the index.
  // CHECK-DAG: note: the body assigns the index too
  for (int i = 0; i < n; i++) {
    s += x[i];
    i++;
  }

  // Counted, but the reverse sweep cannot read the start back: the function
  // writes the variable it came from.
  // CHECK-DAG: note: the start value is written elsewhere in the function
  for (int i = start; i < n; i++)
    s += x[i];

  return s;
}

// An early return can skip a loop while the reverse sweep still runs, so a
// count recomputed from the bounds would be the count of a loop that never
// ran. That is a property of the function, so it takes one of its own.
double returnsEarly(const double* x, int n) {
  if (n < 0)
    return 0;
  double s = 0;
  // CHECK-DAG: note: the function can return before reaching this loop
  for (int i = 0; i < n; i++)
    s += x[i];
  return s;
}

double f(const double* x, int n) { return headers(x, n) + returnsEarly(x, n); }

int main() {
  auto g = clad::gradient(f, "x");
  (void)g;
  return 0;
}
