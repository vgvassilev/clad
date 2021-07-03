//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to differentiate a functor (an object of class
// type with user defined call operator)
//
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -x c++ -std=c++11 Functor.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -x c++ -std=c++11 Functor.cpp

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

class Equation {
  double m_x, m_y;

  public:
  Equation(double x, double y) : m_x(x), m_y(y) {}
  double operator()(double i, double j) {
    return m_x*i*j; + m_y*i*j;
  }
};

int main() {
  Equation E(3, 5);
  
  // Functor object can be passed in 2 ways:

  // 1) by reference
  auto d_E = clad::differentiate(E, "i");

  // 2) as pointer
  auto d_E_pointer = clad::differentiate(&E, "i");

  // no need to explicitly pass functor object as the first argument.
  double res1 = d_E.execute(7, 9);
  double res2 = d_E_pointer.execute(7, 9);

  printf("%.2f %.2f", res1, res2);
}