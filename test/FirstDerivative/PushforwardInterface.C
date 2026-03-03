// RUN: %cladclang %s -I%S/../../include -oForwardModeInterface.out 2>&1 | %filecheck %s
// RUN: ./ForwardModeInterface.out

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// CHECK-NOT: {{error|warning}}

double f(double x) { return x * x; }

double g(double x, double y) { return x * y + x; }

int main() {
  // Demonstrate clad::pushforward API usage
  auto f_pushforward = clad::pushforward(f);
  
  // Call pushforward with seed value 1.0
  auto result = f_pushforward(3.0, 1.0);
  printf("f(3.0) = %f, pushforward = %f\n", result.value, result.pushforward);
  
  // Expected: f(3) = 9, f'(3) * 1 = 6 * 1 = 6
  if (std::abs(result.value - 9.0) > 1e-10) {
    printf("ERROR: Expected f(3.0) = 9.0, got %f\n", result.value);
    return 1;
  }
  if (std::abs(result.pushforward - 6.0) > 1e-10) {
    printf("ERROR: Expected pushforward = 6.0, got %f\n", result.pushforward);
    return 2;
  }
  
  // Test with multiple parameters
  auto g_pushforward = clad::pushforward(g);
  
  // Call pushforward with seed value for x = 2.0, and seed for y = 0.0
  // This computes the directional derivative in the x direction
  auto result2 = g_pushforward(2.0, 3.0, 2.0, 0.0);
  printf("g(2.0, 3.0) = %f, pushforward = %f\n", result2.value, result2.pushforward);
  
  // Expected: g(2, 3) = 2*3 + 2 = 8, dg/dx * 2 + dg/dy * 0 where dg/dx = y+1 = 4, dg/dy = x = 2
  // So pushforward = 4 * 2 + 2 * 0 = 8
  if (std::abs(result2.value - 8.0) > 1e-10) {
    printf("ERROR: Expected g(2.0, 3.0) = 8.0, got %f\n", result2.value);
    return 3;
  }
  if (std::abs(result2.pushforward - 8.0) > 1e-10) {
    printf("ERROR: Expected pushforward = 8.0, got %f\n", result2.pushforward);
    return 4;
  }

  printf("All tests passed!\n");
  return 0;
}
