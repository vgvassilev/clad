//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to differentiate a functor (an object of class
// type with user defined call operator)
//
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang++  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -std=c++11 Templates.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang++  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -std=c++11 Templates.cpp

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

template <class T> class Equation {
  T m_x, m_y;

public:
  Equation(T x = 0, T y = 0) : m_x(x), m_y(y) {}
  T operator()(T i, T j) { return m_x * i * i + m_y * j * j; }
};

template <> class Equation<long double> {
  long double m_x, m_y;

public:
  Equation(long double x = 0, long double y = 0) : m_x(x), m_y(y) {}
  long double operator()(long double i, long double j) {
    return 2 * m_x * i * i + 2 * m_y * j * j;
  }
};

template<class T>
T kinetic_energy(T mass, T velocity) {
  return 0.5*mass*velocity*velocity;
}

int main() {
  Equation<double> E_double(3, 5);
  Equation<long double> E_long_double(3, 5);

  // Differentiating template objects is exactly same as differentiating
  // ordinary objects.
  auto d_E_double = clad::differentiate(E_double, "i");
  auto d_E_long_double = clad::differentiate(E_long_double, "j");
  // calculate derivative of `E_double` wrt `i` when (i, j) = (7, 9) 
  auto E_double_d_i = d_E_double.execute(7, 9);
  // calculate derivative of `E_long_double` wrt `j` when (i, j) = (7, 9)
  auto E_long_double_d_j = d_E_long_double.execute(7, 9);
  
  // Differentiating template functions is also very similar to differentiating
  // ordinary functions.
  auto d_kinetic_energy_double = clad::differentiate(kinetic_energy<double>, "mass");
  auto d_kinetic_energy_long_double = clad::differentiate(kinetic_energy<long double>, "velocity");
  // calculate derivative of `kinetic_energy<double>` wrt `mass` when (mass, velocity) = (7, 9)
  auto kinetic_energy_d_mass = d_kinetic_energy_double.execute(3, 5);
  // calculate derivative of `kinetic_energy<long double>` wrt `velocity` when (mass, velocity) = (7, 9)
  auto kinetic_energy_d_velocity = d_kinetic_energy_long_double.execute(3, 5);
}
