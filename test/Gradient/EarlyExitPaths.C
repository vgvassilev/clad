// RUN: %cladclang %s -I%S/../../include -oEarlyExitPaths.out
// RUN: ./EarlyExitPaths.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oEarlyExitPaths.out
// RUN: ./EarlyExitPaths.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

// The else branch exits. The final iteration must not reverse the multiply
// that follows it: that iteration never stored a value on its restore tape.
double elseReturn(double x) {
  double s = 1;
  for (int i = 0; i < 4; ++i) {
    if (i < 2) {
      s += x;
    } else {
      return s;
    }
    s *= x;
  }
  return s;
}

// Both branches exit the loop body, through different control-flow paths.
// No iteration executes the final multiply, so its reverse must never run.
double bothBranchesExit(double x) {
  double s = 0;
  for (int i = 0; i < 4; ++i) {
    s += x;
    if (i < 2) {
      continue;
    } else {
      return s;
    }
    s *= x;
  }
  return s;
}

// A break in an inner loop does not belong to the returning outer loop.
double nestedLoopBreak(double x) {
  double s = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (j == 1)
        break;
      s += x;
    }
    if (i == 1)
      return s;
    s *= x;
  }
  return s;
}

// A break inside a switch exits the switch, not its enclosing loop. It must
// not cause the return to push an outer-loop break/continue case.
double switchBreak(double x) {
  double s = 0;
  for (int i = 0; i < 3; ++i) {
    switch (i) {
    case 0:
      s += x;
      break;
    default:
      s += 2 * x;
      break;
    }
    if (i == 1)
      return s;
    s *= x;
  }
  return s;
}

// The range-for handler also needs a case for a returning iteration when
// another iteration continues. Exercise both the returning and normal paths.
double rangeReturnContinue(double x, int stop) {
  int values[] = {0, 1, 2, 3};
  double s = 0;
  for (int i : values) {
    if (i == stop)
      return s;
    if (i == 0)
      continue;
    s += x;
  }
  return s;
}

int main() {
  double dx = 0;
  INIT_GRADIENT(elseReturn);
  // (1 + x) * x * x + x * x = x^3 + 2*x^2.
  TEST_GRADIENT(elseReturn, 1, 2, &dx); // CHECK-EXEC: {20.00}

  dx = 0;
  INIT_GRADIENT(bothBranchesExit);
  TEST_GRADIENT(bothBranchesExit, 1, 2, &dx); // CHECK-EXEC: {3.00}

  dx = 0;
  INIT_GRADIENT(nestedLoopBreak);
  // x*x + x.
  TEST_GRADIENT(nestedLoopBreak, 1, 2, &dx); // CHECK-EXEC: {5.00}

  dx = 0;
  INIT_GRADIENT(switchBreak);
  // x*x + 2*x.
  TEST_GRADIENT(switchBreak, 1, 2, &dx); // CHECK-EXEC: {6.00}

  auto rangeGrad = clad::gradient(rangeReturnContinue, "x");
  dx = 0;
  rangeGrad.execute(2, 2, &dx);
  printf("range early: %.2f\n", dx); // CHECK-EXEC: range early: 1.00
  dx = 0;
  rangeGrad.execute(2, 4, &dx);
  printf("range normal: %.2f\n", dx); // CHECK-EXEC: range normal: 3.00
}
