//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to differentiate a functor (an object of class
// type with user defined call operator)
//
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang++  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -std=c++11 Functor.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang++  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -std=c++11 Functor.cpp

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

// A class type with user-defined call operator
class Equation {
  double m_x, m_y;

  public:
    Equation(double x = 0, double y = 0) : m_x(x), m_y(y) {}
    double operator()(double i, double j) { return m_x * i * j + m_y * i * j; }
    void setX(double x) { m_x = x; }
};

int main() {
  Equation E(3, 5);

  // Functor is an object of any type which have user defined call operator.
  //
  // Clad differentiation functions can directly differentiate functors.
  // Functors can be passed to clad differentiation functions in two distinct
  // ways:

  // 1) Pass by reference
  // differentiates `E` wrt parameter `i`
  // object `E` is saved in the `CladFunction` object `d_E`
  auto d_E = clad::differentiate(E, "i");

  // 2) Pass as pointers
  // differentiates `E` wrt parameter `i`
  // object `E` is saved in the `CladFunction` object `d_E_pointer`
  auto d_E_pointer = clad::differentiate(&E, "i");

  // calculate differentiation of `E` when (i, j) = (7, 9)
  double res1 = d_E.execute(7, 9);
  double res2 = d_E_pointer.execute(7, 9);
  printf("%.2f %.2f\n", res1, res2);

  Equation AnotherE(11, 13);

  // If differentiation of `E` is not needed, then `d_E` and `d_E_pointer` can
  // be reused for computing differentiation of another functor of the same
  // type by modifying the saved object in `CladFunction`. This can save
  // computation time.
  d_E.setObject(&AnotherE);
  d_E_pointer.setObject(&AnotherE);

  // calculate differentiation of `AnotherE` when (i, j) = (7, 9)
  res1 = d_E.execute(7, 9);
  res2 = d_E_pointer.execute(7, 9);
  printf("%.2f %.2f\n", res1, res2);

  // Differentiation of any other functor of the same type can also be computed
  // without changing the saved object in `CladFunction` object by explicitly
  // passing the functor as the first argument while calling
  // `CladFunction::execute`.
  // Calculates differentiation of `E` when (i, j) = (7, 9)
  res1 = d_E.execute(E, 7, 9);
  res2 = d_E_pointer.execute(E, 7, 9);
  printf("%.2f %.2f\n", res1, res2);

  // Saved object in `CladFunction` object can be removed by using
  // `CladFunction::clearObject`
  d_E.clearObject();
  d_E_pointer.clearObject();

  // Now it is necessary to explicitly pass the functor object while calling
  // `CladFunction::execute`, since no saved object is available.
  // Calculates differentiation of `E` when (i, j) = (7, 9)
  res1 = d_E.execute(E, 7, 9);
  // Calculates differentiation of `AnotherE` when (i, j) = (7, 9)
  res2 = d_E_pointer.execute(AnotherE, 7, 9);
  printf("%.2f %.2f\n", res1, res2);
}
