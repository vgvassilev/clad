// RUN: %cladclang -I%S/../../include %s 2>&1 | %filecheck %s

#include <algorithm>
#include <cstddef>

#include "clad/Differentiator/Differentiator.h"

unsigned int rawBinNumber(double x, const double* boundaries,
                          std::size_t nBoundaries) {
  const double* end = boundaries + nBoundaries;
  const double* it = std::lower_bound(boundaries, end, x);
  while (boundaries != it && (end == it || end == it + 1 || x < *it))
    --it;
  return it - boundaries;
}

double roo_codegen_0(double* params, const double* obs, const double* xlArr) {
  double out = 0.;
  double t23[5]{1. + params[0], 1. + params[1], 1. + params[2],
                1. + params[3], 1. + params[4]};
  for (int i = 0; i < 5; ++i) {
    const double t215 = t23[rawBinNumber(obs[i], xlArr, 6)];
    out += t215;
  }
  return out;
}

int main() {
  auto grad = clad::gradient(roo_codegen_0, "params");
  (void)grad;
}

// CHECK: void roo_codegen_0_grad_0(double *params, const double *obs, const double *xlArr, double *_d_params) {
// CHECK-NOT: rawBinNumber_pullback
