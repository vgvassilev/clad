// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-overloads
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double area(double r) { return 3.0 * r * r; }
double area(double w, double h) { return w * h; }

int main() {
  auto d_disk = clad::gradient(static_cast<double (*)(double)>(area));
  auto d_rect = clad::gradient(static_cast<double (*)(double, double)>(area));

  double d_r = 0, d_w = 0, d_h = 0;
  d_disk.execute(2, &d_r);
  d_rect.execute(3, 5, &d_w, &d_h);
  std::cout << d_r << " " << d_w << " " << d_h << "\n"; // prints: 12 5 3
}
// docs-end-overloads
