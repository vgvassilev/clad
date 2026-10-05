// RUN: %cladclang -O0 %s -I%S/../../include -o %t.default > %t.default.codegen
// RUN: %t.default | %filecheck_exec %s
// RUN: %cladclang -O3 -DNDEBUG %s -I%S/../../include -o %t.optimized > %t.optimized.codegen
// RUN: %t.optimized | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "../../benchmark/ActivityWasteKernels.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

double dummy_out = 0;

// Keep the numerical checks active in optimized builds with NDEBUG.
static void check(const char* label, double actual, double expected) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > 1e-12 * (1 + std::fabs(expected))) {
    std::fprintf(stderr, "%s: got %.17g, expected %.17g\n", label, actual,
                 expected);
    std::exit(1);
  }
}

int main() {
  auto positive = clad::gradient(positive_control, "x");
  auto interproc = clad::gradient(interproc_passive, "x");
  auto field = clad::gradient(dead_struct_field, "x");
  auto array = clad::gradient(dead_array_chain, "x");
  auto branch = clad::gradient(branch_guarded, "x");
  auto useful = clad::gradient(useful_not_varied, "x");
  std::array<double, 4> x = {-0.5, 0.0, 0.25, 0.5};
  std::array<double, 4> dx{};

  for (int n : {1, 4}) {
    // Repeat with reset state to catch cross-call contamination.
    for (int repeat = 0; repeat < 2; ++repeat) {
      dx.fill(0);
      dummy_out = 0;
      positive.execute(x.data(), n, dx.data());
      for (int i = 0; i < n; ++i)
        check("positive_control", dx[i], 4 * x[i] * x[i] * x[i]);

      for (int work : {0, 1, 4, 17}) {
        dx.fill(0);
        dummy_out = 0;
        interproc.execute(x.data(), n, work, dx.data());
        for (int i = 0; i < n; ++i)
          check("interproc_passive", dx[i], 1);
      }

      dx.fill(0);
      dummy_out = 0;
      field.execute(x.data(), n, dx.data());
      for (int i = 0; i < n; ++i)
        check("dead_struct_field", dx[i], 1 + 2 * x[i]);

      dx.fill(0);
      dummy_out = 0;
      array.execute(x.data(), n, dx.data());
      for (int i = 0; i < n; ++i)
        check("dead_array_chain", dx[i], i == 0 ? 2 * x[0] : 0);

      for (int flag : {0, 1}) {
        dx.fill(0);
        dummy_out = 0;
        branch.execute(x.data(), n, flag, dx.data());
        for (int i = 0; i < n; ++i)
          check("branch_guarded", dx[i], i == 0 ? 2 * x[0] : 0);
      }
    }
  }

  // Exercise the scratch buffer's upper bound without a large benchmark loop.
  std::vector<double> large_x(kMaxN, 0.5);
  std::vector<double> large_dx(kMaxN, 0);
  dummy_out = 0;
  array.execute(large_x.data(), kMaxN, large_dx.data());
  for (int i = 0; i < kMaxN; ++i)
    check("dead_array_chain maximum", large_dx[i], i == 0 ? 1 : 0);

  for (double value : x) {
    for (int reps : {0, 1, 2, 3, 4, 17}) {
      for (int repeat = 0; repeat < 2; ++repeat) {
        double derivative = 0;
        useful.execute(value, reps, &derivative);
        check("useful_not_varied", derivative, 1);
      }
    }
  }
  std::puts("ActivityWaste gradients OK");
  // CHECK-EXEC: ActivityWaste gradients OK
}
