// RUN: %cladclang %s -I%S/../../include -oPushforwardInterface.out 2>&1 | %filecheck %s
// RUN: ./PushforwardInterface.out

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cstdio>

// CHECK-NOT: {{error|warning}}

double f(double x) { return x * x; }

double g(double x, double y) { return x * y + x; }

double h(double x, double y, double z) { return x * y * z + x * x; }

float f_float(float x) { return x * x * x; }

int test_count = 0;
int fail_count = 0;

bool approx_equal(double a, double b, double tol = 1e-10) {
  return std::fabs(a - b) < tol;
}

void assert_test(bool condition, const char* test_name) {
  test_count++;
  if (!condition) {
    printf("FAIL: %s\n", test_name);
    fail_count++;
  }
}

int main() {
  // Single-parameter function: f(x) = x^2
  {
    auto f_pushforward = clad::pushforward(f);
    auto result = f_pushforward(3.0, 1.0);
    assert_test(approx_equal(result.value, 9.0), "f(x): value");
    assert_test(approx_equal(result.pushforward, 6.0), "f(x): pushforward");
  }

  // Directional derivative scaling
  {
    auto f_pushforward = clad::pushforward(f);
    auto result = f_pushforward(3.0, 2.0);
    assert_test(approx_equal(result.value, 9.0), "f(x): scaled pushforward");
    assert_test(approx_equal(result.pushforward, 12.0), "f(x): seed=2.0");
  }

  // Zero seed
  {
    auto f_pushforward = clad::pushforward(f);
    auto result = f_pushforward(3.0, 0.0);
    assert_test(approx_equal(result.value, 9.0), "f(x): zero seed value");
    assert_test(approx_equal(result.pushforward, 0.0), "f(x): zero seed derivative");
  }

  // Negative input
  {
    auto f_pushforward = clad::pushforward(f);
    auto result = f_pushforward(-2.0, 1.0);
    assert_test(approx_equal(result.value, 4.0), "f(x): negative input");
    assert_test(approx_equal(result.pushforward, -4.0), "f(x): negative derivative");
  }

  // Two-parameter function: g(x,y) = xy + x, directional derivatives
  {
    auto g_pushforward = clad::pushforward(g);
    // seed in x direction
    auto result = g_pushforward(2.0, 3.0, 2.0, 0.0);
    assert_test(approx_equal(result.value, 8.0), "g(x,y): value");
    assert_test(approx_equal(result.pushforward, 8.0), "g(x,y): x-direction");
    // seed in y direction
    result = g_pushforward(2.0, 3.0, 0.0, 1.0);
    assert_test(approx_equal(result.pushforward, 2.0), "g(x,y): y-direction");
    // mixed seed
    result = g_pushforward(2.0, 3.0, 1.0, 1.0);
    assert_test(approx_equal(result.pushforward, 6.0), "g(x,y): mixed seed");
  }

  // Three-parameter function: h(x,y,z) = xyz + x^2
  {
    auto h_pushforward = clad::pushforward(h);
    auto result = h_pushforward(2.0, 3.0, 2.0, 1.0, 0.0, 0.0);
    assert_test(approx_equal(result.value, 16.0), "h(x,y,z): value");
    assert_test(approx_equal(result.pushforward, 10.0), "h(x,y,z): seed (1,0,0)");
  }

  // Float specialization: f_float(x) = x^3
  {
    auto f_float_pushforward = clad::pushforward(f_float);
    auto result = f_float_pushforward(2.0f, 1.0f);
    assert_test(approx_equal((double)result.value, 8.0, 1e-6f), "f_float: value");
    assert_test(approx_equal((double)result.pushforward, 12.0, 1e-6f), "f_float: pushforward");
  }

  // Critical point
  {
    auto f_pushforward = clad::pushforward(f);
    auto result = f_pushforward(0.0, 1.0);
    assert_test(approx_equal(result.value, 0.0), "f(0): value");
    assert_test(approx_equal(result.pushforward, 0.0), "f(0): derivative");
  }

  return fail_count == 0 ? 0 : 1;
}
