// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-functor
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

// A class type with user-defined call operator.
class Equation {
  double m_x, m_y;

public:
  Equation(double x = 0, double y = 0) : m_x(x), m_y(y) {}
  double operator()(double i, double j) { return m_x * i * j + m_y * i * j; }
  void setX(double x) { m_x = x; }
};

int main() {
  Equation E(3, 5);

  // A functor is an object of any type which has a user defined call
  // operator.
  //
  // Clad differentiation functions can directly differentiate functors.
  // Functors can be passed to clad differentiation functions in two distinct
  // ways:

  // 1) Pass by reference.
  // Differentiates 'E' with respect to parameter 'i'. The object 'E' is saved
  // in the 'CladFunction' object 'd_E'.
  auto d_E = clad::differentiate(E, "i");

  // 2) Pass as pointers.
  // Differentiates 'E' with respect to parameter 'i'. The object 'E' is saved
  // in the 'CladFunction' object 'd_E_pointer'.
  auto d_E_pointer = clad::differentiate(&E, "i");

  // Calculate the derivative of 'E' when (i, j) = (7, 9).
  std::cout << d_E.execute(7, 9) << "\n";         // prints: 72
  std::cout << d_E_pointer.execute(7, 9) << "\n"; // prints: 72
}
// docs-end-functor
