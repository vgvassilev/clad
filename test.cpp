#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double f(double x, double y) { return x + y; }


int main() {
  auto ff = clad::gradient(f);

  ff.dump();

  double dx = 0, dy = 0;

  ff.execute(1, 1, &dx, &dy);

  std::cout << dx << " " << dy << std::endl;
}

