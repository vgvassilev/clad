// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-array-hessian
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double h(double x, double arr[3]) { return x * arr[0] * arr[1] * arr[2]; }

int main() {
  // Differentiating h w.r.t all the input variables (x, arr).
  // Note that the array and the indexes are explicitly mentioned even though
  // all the indexes (0, 1 and 2) are being differentiated.
  auto h_hess = clad::hessian(h, "x, arr[0:2]");

  double x = 2, arr[3] = {1, 2, 3};

  // Create memory for the hessian matrix. The minimum required size of the
  // matrix is the square of the number of independent variables. Since there
  // are 3 indexes of the array and a scalar variable, the total number of
  // independent variables is 4. clad adds into the matrix, so it starts at
  // zero.
  double mat[16] = {0};

  // The inputs to the original function h (i.e x and arr) are passed
  // followed by the output matrix.
  h_hess.execute(x, arr, mat);

  printf("hessian matrix: \n"
         "{ %g, %g, %g, %g\n"
         "  %g, %g, %g, %g\n"
         "  %g, %g, %g, %g\n"
         "  %g, %g, %g, %g }\n",
         mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7],
         mat[8], mat[9], mat[10], mat[11], mat[12], mat[13], mat[14], mat[15]);
  // prints: hessian matrix:
  // prints: { 0, 6, 3, 2
  // prints:   6, 0, 6, 4
  // prints:   3, 6, 0, 2
  // prints:   2, 4, 2, 0 }
}
// docs-end-array-hessian
