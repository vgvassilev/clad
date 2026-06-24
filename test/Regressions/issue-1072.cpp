// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -I%S/../../include %s | %filecheck %s
// RUN: %cladclang -I%S/../../include %s | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

#include <vector>

double fn(double u) {
  std::vector<double> vec;
  vec.push_back(u);
  return vec[0];
}

// CHECK: ::size_type _r0

int main() {
  auto grad = clad::gradient(fn);
}

