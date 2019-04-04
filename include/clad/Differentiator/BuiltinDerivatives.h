//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_BUILTIN_DERIVATIVES
#define CLAD_BUILTIN_DERIVATIVES

// Avoid assertion custom_derivative namespace not found. FIXME: This in future
// should go.
namespace custom_derivatives{}

#include <cmath>

namespace custom_derivatives {
  // define functions' derivatives from std
  namespace std {
    // There are 4 overloads:
    // float       sin( float arg );
    // double      sin( double arg );
    // long double sin( long double arg );
    // double      sin( Integral arg ); (since C++11)
    template<typename R, typename A> R sin(A x) {
      return (R)1;
      //return (R)::std::cos((A)x);
    }

    // There are 4 overloads:
    // float       cos( float arg );
    // double      cos( double arg );
    // long double cos( long double arg );
    // double      cos( Integral arg ); (since C++11)
    template<typename R, typename A> R cos(A x) {
      return (R)1;
      //return (R)-::std::sin((A)x);
    }

    // There are 4 overloads:
    // float       sqrt( float arg );
    // double      sqrt( double arg );
    // long double sqrt( long double arg );
    // double      sqrt( Integral arg ); (since C++11)
    //template<typename R, typename A> R sqrt(A x) {
    //  return (R)(((A)1)/(2*((R)std::sqrt((A)x))));
    //}
  }// end namespace std

  template<typename T>
  T exp_darg0(T x) {
    return exp(x);
  }

  template<typename T>
  T sin_darg0(T x) {
    return cos(x);
  }

  template<typename T>
  T cos_darg0(T x) {
    return (-1) * sin(x);
  }

  template<typename T>
  T sqrt_darg0(T x) {
     return ((T)1)/(((T)2)*sqrt(x));
  }

#ifdef MACOS
  float sqrtf_darg0(float x) {
    return 1.F/(2.F*sqrtf(x));
  }
#endif

  double pow_darg0(double x, double exponent) {
    return exponent * ::std::pow(x, exponent-1);
  }

  double pow_darg1(double x, double exponent) {
    return ::std::pow(x, exponent) * ::std::log(x);
  }

  void pow_grad(double x, double exponent, double* result) {
    result[0] += pow_darg0(x, exponent);
    result[1] += pow_darg1(x, exponent);
  }

} // end namespace builtin_derivatives

#endif //CLAD_BUILTIN_DERIVATIVES
