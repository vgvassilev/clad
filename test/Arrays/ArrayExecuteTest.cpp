#include "clad/Differentiator/Differentiator.h"
#include <iostream>
#include <cassert>

struct SquareFunctor {
  double execute(double* args, unsigned long size) {
    if (size == 0) return 0.0;
    return args[0] * args[0]; 
  }
};

int main() {
  SquareFunctor f;
  double arr[3] = {2.0, 3.0, 4.0};

  double result = clad::execute(f, arr);

  std::cout << "Result: " << result << std::endl;
  
  if (result == 4.0) {
      std::cout << "Test Passed!" << std::endl;
      return 0;
  }
  std::cout << "Test Failed!" << std::endl;
  return 1;
}
