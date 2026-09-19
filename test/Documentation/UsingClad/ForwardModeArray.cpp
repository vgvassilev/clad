// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-forward-mode-array
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double fn_arr(double* arr, int n) {
  double res = 0;
  for (int i = 0; i < n - 1; ++i)
    res += arr[i] * arr[i + 1];
  return res;
}

int main() {
  // Differentiate 'fn_arr' with respect to element '1' of the 'arr' parameter.
  auto d_fn_arr = clad::differentiate(fn_arr, "arr[1]");
  double arr[5] = {1, 2, 3, 4, 5};
  std::cout << d_fn_arr.execute(arr, 5) << "\n"; // prints: 4
}
// docs-end-forward-mode-array
