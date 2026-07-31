// RUN: %cladclang %s -I%S/../../include -oReallocShrink.out 2>&1 | %filecheck %s
// RUN: ./ReallocShrink.out | %filecheck_exec %s

// Shrinking in-place realloc: the reverse sweep must resize the buffer (and its
// adjoint) back to the pre-realloc size so the pre-realloc reads stay in bounds
// and the adjoint accumulates correctly. Runs Memcheck-clean under the valgrind
// CI row.

#include "clad/Differentiator/Differentiator.h"
#include <cstdlib>

// res = (x + x*x) + x  ->  dres/dx = 2 + 2x ; at x = 3 -> 8
double shrink_realloc(double x) {
  double* p = (double*)malloc(2 * sizeof(double));
  p[0] = x;
  p[1] = x * x;
  double res = p[0] + p[1];
  p = (double*)realloc(p, 1 * sizeof(double)); // shrink 2 -> 1
  res += p[0];
  free(p);
  return res;
}

// Two in-place reallocs on the same pointer, so each must restore the size the
// buffer had before *it*: res = (x + x*x + x*x*x) + x + x -> x^3 + x^2 + 3x;
// dres/dx = 3*x*x + 2*x + 3; at x = 2 -> 19.
double chained_realloc(double x) {
  double* p = (double*)malloc(3 * sizeof(double));
  p[0] = x;
  p[1] = x * x;
  p[2] = x * x * x;
  double res = p[0] + p[1] + p[2];
  p = (double*)realloc(p, 1 * sizeof(double)); // shrink 3 -> 1
  res += p[0];
  p = (double*)realloc(p, 2 * sizeof(double)); // grow  1 -> 2
  p[1] = x;
  res += p[1];
  free(p);
  return res;
}

// realloc(NULL, n) grows an empty buffer (there is no prior allocation to size,
// so no shadow is tracked and the reverse sweep keeps the reallocated pointer,
// which is sound for a grow). res = x + x*x -> dres/dx = 1 + 2x; at x = 3 -> 7.
double realloc_from_null(double x) {
  double* p = nullptr;
  p = (double*)realloc(p, 2 * sizeof(double));
  p[0] = x;
  p[1] = x * x;
  double res = p[0] + p[1];
  free(p);
  return res;
}

int main() {
  auto g = clad::gradient<clad::opts::disable_tbr>(shrink_realloc, "x");
  double dx = 0;
  g.execute(3.0, &dx);
  printf("{%.2f}\n", dx); // CHECK-EXEC: {8.00}

  auto gn = clad::gradient<clad::opts::disable_tbr>(realloc_from_null, "x");
  double dxn = 0;
  gn.execute(3.0, &dxn);
  printf("{%.2f}\n", dxn); // CHECK-EXEC: {7.00}

  auto gc = clad::gradient<clad::opts::disable_tbr>(chained_realloc, "x");
  double dxc = 0;
  gc.execute(2.0, &dxc);
  printf("{%.2f}\n", dxc); // CHECK-EXEC: {19.00}
}
