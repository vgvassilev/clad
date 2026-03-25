// RUN: %cladclang -std=c++20 -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s
// UNSUPPORTED: clang-10, clang-11, clang-12, clang-13, clang-14, clang-15, clang-16
// XFAIL: valgrind

#include <iostream>
#include <functional>
#include <thread>

#include "clad/Differentiator/Differentiator.h"

void add_square(double x, double& out) {
  out = x * x;
}

std::reference_wrapper<double> wrap(double& out) { return std::ref(out); }

namespace custom {
std::reference_wrapper<double> ref(double& out) { return std::ref(out); }
} // namespace custom

double f(double x, double y) {
  double rx = 0.0;
  double ry = 0.0;
  std::thread t1(add_square, x, std::ref(rx));
  std::thread t2(add_square, y, std::ref(ry));
  t1.join();
  t2.join();
  return rx + ry;
}

double g(double x, double y) { return x * x + y * y; }

double f_non_ref_wrapper(double x) {
  double rx = 0.0;
  std::thread t(add_square, x, wrap(rx));
  t.join();
  return rx;
}

double f_custom_ref(double x) {
  double rx = 0.0;
  std::thread t(add_square, x, custom::ref(rx));
  t.join();
  return rx;
}

int main() {
  auto df = clad::differentiate(f, "x");
  std::cout << "f'(3,4) = " << df.execute(3.0, 4.0) << "\n";
  auto gf = clad::gradient(g);
  double dx = 0.0, dy = 0.0;
  gf.execute(3.0, 4.0, &dx, &dy);
  std::cout << "grad g(3,4) = (" << dx << ", " << dy << ")\n";

  auto dnonref = clad::differentiate(f_non_ref_wrapper, "x");
  std::cout << "f_non_ref_wrapper'(3) = " << dnonref.execute(3.0) << "\n";

  auto dcustomref = clad::differentiate(f_custom_ref, "x");
  std::cout << "f_custom_ref'(3) = " << dcustomref.execute(3.0) << "\n";

  // CHECK-EXEC: f'(3,4) = 6
  // CHECK-EXEC: grad g(3,4) = (6, 8)
  // CHECK-EXEC: f_non_ref_wrapper'(3) = 6
  // CHECK-EXEC: f_custom_ref'(3) = 6
  return 0;
}
