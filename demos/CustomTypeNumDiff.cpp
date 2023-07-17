//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how users can numerically differentiate functions that
// take user-defined types as parameters using clad.
//
// author: Garima Singh
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang++  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -std=c++11 CustomTypeNumDiff.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang++  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -std=c++11 CustomTypeNumDiff.cpp

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

#include <iostream> // for std::* functions.

// A user defined class to emulate fixed point decimal numbers.
// It stores the significand and inverted exponent as integers.
// for a number of the form ax10e-b
// 'number' refers to 'a' and 'scale' refers to 'b'.
class FixedPointFloat {
  long int number;
  unsigned scale;

public:
  FixedPointFloat(unsigned num, unsigned n) {
    // If there are any trailing zeros to num and a non-zero scale value
    // normalize the number so that we do not do unnecessary multiplications.
    while (!num % 10 && n) {
      num = std::floor(num / 10);
      n--;
    }
    number = num;
    scale = n;
  }
  // A function to get the double equivalent of the decimal number stored.
  double getDouble(unsigned num, unsigned scale) {
    return (double)num / std::pow(10, scale);
  }
  // A function to scale the value of a given number, we use this to
  // facilitate evaluation of different operations on this class.
  unsigned getScaled(unsigned num, unsigned scale) {
    for (unsigned i = 0; i < scale; i++) {
      num *= 10;
    }
    return num;
  }
  // Gets the double representation of the number represented by the
  // calling object.
  double getNum() { return getDouble(number, scale); }
  // Some operator overloads for this class...
  FixedPointFloat operator+(FixedPointFloat a) {
    unsigned xa, yb;
    // Here, let us understand this with an example.
    // consider two numbers -> 0.005 and 0.00002 ~ 5x10e-3 and 2x10e-5
    // converting them to FixedPointFloat will give us a pair like so:
    // (5, 3) and (2, 5)
    // To add the above numbers, we follow the following...
    // Get the max scale value (i.e the minimum exponent value)
    // in our case, it is 5.
    unsigned maxScale = std::max(a.scale, scale);
    // Scale both numbers with the remaining scale value.
    // essentially, do the following:
    // (5x10e-3 + 2x10e-5) * (10e5/10e5)
    // (5x10e(-3 + 5) + 2x10e(-5+5)) * (1/10e5)
    // => (500 + 2) * (1/10e5)
    xa = getScaled(a.number, maxScale - a.scale); // = 500
    yb = getScaled(number, maxScale - scale);     // = 2
    // => (500 + 2) * (1/10e5) = 502x10e-5
    return FixedPointFloat(xa + yb, maxScale); // = (502, 5)
  }
  // Similar to above, but here, we just subtract the number instead of adding
  // it.
  FixedPointFloat operator-(FixedPointFloat a) {
    unsigned xa, yb;
    unsigned maxScale = std::max(a.scale, scale);
    xa = getScaled(a.number, maxScale - a.scale);
    yb = getScaled(number, maxScale - scale);
    return FixedPointFloat(xa - yb, maxScale);
  }

  FixedPointFloat operator*(FixedPointFloat a) {
    unsigned xa, yb;
    // This is fairly straight forward. Let us take the same example as before
    // 0.005 and 0.00002 ~ 5x10e-3 and 2x10e-5
    // This operation is equivalent to
    // = 5x10e-3 x 2x10e-5
    // = (5x2) x 10e(-5-3)
    // = 10x10e-8 (which is eventually reformed to 1x10e-7)
    return FixedPointFloat(a.number * number, a.scale + scale); // = (1, 7)
  }
};

// This function is essential if we want to differentiate a function with
// user-defined data types as arguments. This function describes the "rules"
// of how to differentiate user-defined data types by specifying how to
// introduce a small change (h) in the object of the user-defined data type.
// Details on how to overload this function are provided in the docs.
FixedPointFloat
updateIndexParamValue(FixedPointFloat arg, std::size_t idx, std::size_t currIdx,
                      int multiplier, numerical_diff::precision& h_val,
                      std::size_t n = 0, std::size_t i = 0) {
  if (idx == currIdx) {
    // Here we just introduce an h of 0.000001 to all our FixedPointFloat
    // numbers.
    FixedPointFloat h(1, 5);
    h_val = (h_val == 0) ? h.getNum() : h_val;
    FixedPointFloat Fmultiplier(multiplier, 0);
    return arg + Fmultiplier * h;
  }
  return arg;
}

// A simple multiplication function.
// currently we can only numerically differentiate the following types:
// - all scalar numeric types
// - all types with basic arithmetic operators overloaded.
// - all types that are implicitly convertible to some numeric type.
double func(FixedPointFloat x, FixedPointFloat y) { return (x * y).getNum(); }

int main() {
  // Define some values which will reflect the derivative later.
  double dx = 0, dy = 0;
  // Define our inputs: 3x10e-3 and 7x10e-2 or 0.003 and 0.07.
  FixedPointFloat x(3, 3), y(7, 2);
  // Push the dx and dy values to a tape created by us.
  // This is how we return the derivative with respect to all arguments.
  // The order of being placed in this tape should be the same as the order of
  // the arguments being passed to the function.
  clad::tape<clad::array_ref<
      double /*This should be the return value of the function you want to differentiate.*/>>
      grad = {};
  // Place the l-value reference of the variables in the tape.
  grad.emplace_back(&dx);
  grad.emplace_back(&dy);
  // Finally, call the numerical diff method, keep the order of the arguments to
  // the function in mind!
  numerical_diff::central_difference(func, grad, /*printErrors=*/0, x, y);
  // Finally print the results!
  std::cout << "Result of df/dx is = " << dx << "\nResult of df/dx is = " << dy
            << std::endl;
}
