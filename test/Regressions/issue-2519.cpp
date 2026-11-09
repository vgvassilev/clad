// RUN: %cladclang -std=c++20 -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

struct Point3D {
  double x;
  double y;
  double z;
};

double distance_squared(double u, double v, double w) {
  Point3D p{.x = u * u, .y = v * v, .z = w * w};
  return p.x + p.y + p.z;
}

int main() {
  auto df = clad::differentiate(distance_squared, "u");
  // CHECK-EXEC: df/du: 4
  std::cout << "df/du: " << df.execute(2.0, 3.0, 4.0) << std::endl;

  double g_u = 0, g_v = 0, g_w = 0;
  auto grad = clad::gradient(distance_squared);
  grad.execute(2.0, 3.0, 4.0, &g_u, &g_v, &g_w);
  // CHECK-EXEC: grad: 4 6 8
  std::cout << "grad: " << g_u << " " << g_v << " " << g_w << std::endl;
}
