// RUN: %cladclang -std=c++20 -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s
// UNSUPPORTED: clang-10, clang-11, clang-12, clang-13, clang-14, clang-15, clang-16
// XFAIL: valgrind

#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>
#include <vector>

#include "clad/Differentiator/Differentiator.h"

// Reproducer from issue #1721 (SOFIE-generated pattern)
inline void scalar_prod(float* output, int m, const float* a, const float* b) {
  for (int i = 0; i < m; ++i) {
    output[i] = a[i] * b[0];
  }
}

void inner_func(float const* x, float const* theory_params, float* linear_3) {
  (void)x;
  float val_0[5] = {1.F, 1.F, 1.F, 1.F, 1.F};
  float linear[5] = {0.F, 0.F, 0.F, 0.F, 0.F};

  scalar_prod(linear, 5, val_0, theory_params);

  for (int i = 0; i < 5; ++i) {
    linear_3[0] += linear[i];
  }
}

float my_func(float const* x, float const* theory_params) {
  float out = 0.F;
  inner_func(x, theory_params, &out);
  return out;
}

int main() {
  std::vector<float> input1{5.0F, 2.0F, 1.0F, -1.0F, 1.0F};
  std::vector<float> input2{0.0F};

  auto func = [&](std::span<float> params) {
    return my_func(input1.data(), params.data());
  };

  auto numDiff = [&](int i) {
    const float eps = 1e-4F;
    std::vector<float> p{input2};
    p[i] = input2[i] - eps;
    float funcValDown = func(p);
    p[i] = input2[i] + eps;
    float funcValUp = func(p);
    return (funcValUp - funcValDown) / (2 * eps);
  };

  for (std::size_t i = 0; i < input2.size(); ++i) {
    std::cout << i << ":\n";
    std::cout << " numr : " << numDiff((int)i) << "\n";
  }

  float grad_output[]{0.F, 0.F, 0.F, 0.F, 0.F};
  auto g_grad = clad::gradient(my_func, "theory_params");
  g_grad.execute(input1.data(), input2.data(), grad_output);

  std::fill(std::begin(grad_output), std::end(grad_output), 0.F);
  g_grad.execute(input1.data(), input2.data(), grad_output);

  std::cout << " clad : " << grad_output[0] << "\n";

  // CHECK-EXEC: 0:
  // CHECK-EXEC: numr : 5
  // CHECK-EXEC: clad : 5
  return 0;
}
