#include <iostream>
#include <memory>


#include "clad/Differentiator/STLBuiltins.h"

void test_unique_ptr() {
  // Create a unique_ptr managing a double value (3.0).
  auto up = std::make_unique<double>(3.0);
  // Create a unique_ptr for the derivative, initially set to 0.
  auto d_up = std::make_unique<double>(0.0);


  auto va_unique = clad::custom_derivatives::class_functions::operator_star_reverse_forw(&up, &d_up);
  
  // For f(x)=x^2, the derivative is 2*x (expected: 6 when x == 3).
  double x = va_unique.value;
  double pullback = 2.0 * x;
  
  clad::custom_derivatives::class_functions::operator_star_pullback(&up, pullback, &d_up);

  std::cout << "Unique_ptr Test:" << std::endl;
  std::cout << "Value: " << *up << std::endl;
  std::cout << "Derivative: " << *d_up << std::endl;
}

void test_shared_ptr() {
  // Create a shared_ptr managing a double value (3.0).
  auto sp = std::make_shared<double>(3.0);
  // Create a shared_ptr for the derivative, initially set to 0.
  auto d_sp = std::make_shared<double>(0.0);

  auto va_shared = clad::custom_derivatives::class_functions::operator_star_reverse_forw(&sp, &d_sp);
  
  double x = va_shared.value; // Expected to be 3.0.
  double pullback = 2.0 * x;  // Expected pullback: 6.0.

  clad::custom_derivatives::class_functions::operator_star_pullback(&sp, pullback, &d_sp);

  std::cout << "Shared_ptr Test:" << std::endl;
  std::cout << "Value: " << *sp << std::endl;
  std::cout << "Derivative: " << *d_sp << std::endl;
}

int main() {
  test_unique_ptr();
  std::cout << std::endl;
  test_shared_ptr();
  return 0;
}
