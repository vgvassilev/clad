// RUN: %cladclang %s -I%S/../../include -o%t.out 2>&1
// RUN: %t.out

#include "clad/Differentiator/TapeAndPullback.h"

#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
  clad::tape<double> t = {};
  clad::push<double>(t, 3.0);

  double d_x = 0.0;
  double d_exponent = 0.0;
  clad::custom_derivatives::std::pow_pullback(clad::back<double>(t), 2.0, 1.0,
                                              &d_x, &d_exponent);

  if (std::fabs(d_x - 6.0) > 1e-12) {
    std::printf("unexpected derivative: %f\n", d_x);
    return 1;
  }

  double v = clad::pop<double>(t);
  if (std::fabs(v - 3.0) > 1e-12) {
    std::printf("unexpected tape value: %f\n", v);
    return 2;
  }

  double arr[3] = {1.0, 2.0, 3.0};
  clad::tape<double[3]> arrTape = {};
  clad::push(arrTape, arr);
  auto& arrBack = clad::back(arrTape);
  if (std::fabs(arrBack[0] - 1.0) > 1e-12 ||
      std::fabs(arrBack[1] - 2.0) > 1e-12 ||
      std::fabs(arrBack[2] - 3.0) > 1e-12) {
    std::printf("unexpected array tape values\n");
    return 3;
  }
  clad::pop(arrTape);

  clad::tape<int, 64, 1024, true> mtTape = {};
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i) {
    workers.emplace_back([&]() {
      for (int j = 0; j < 1000; ++j)
        clad::push<int>(mtTape, 1);
    });
  }
  for (auto& worker : workers)
    worker.join();

  if (mtTape.size() != 4000) {
    std::printf("unexpected multithreaded tape size: %zu\n", mtTape.size());
    return 4;
  }

  return 0;
}
