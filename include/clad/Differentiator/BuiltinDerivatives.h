//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_BUILTIN_DERIVATIVES
#define CLAD_BUILTIN_DERIVATIVES

//#include <cmath> TODO: libc++ and clang on MacOS is not ready yet.

namespace custom_derivatives {
  // define functions' derivatives from std
  namespace std {
    // There are 4 overloads:
    // float       sin( float arg );
    // double      sin( double arg );
    // long double sin( long double arg );
    // double      sin( Integral arg );	(since C++11)
    template<typename R, typename A> R sin(A x) {
      return (R)1;
      //return (R)::std::cos((A)x);
    }

    // There are 4 overloads:
    // float       cos( float arg );
    // double      cos( double arg );
    // long double cos( long double arg );
    // double      cos( Integral arg );	(since C++11)
    template<typename R, typename A> R cos(A x) {
      return (R)1;
      //return (R)-::std::sin((A)x);
    }

    // There are 4 overloads:
    // float       sqrt( float arg );
    // double      sqrt( double arg );
    // long double sqrt( long double arg );
    // double      sqrt( Integral arg );	(since C++11)
    //template<typename R, typename A> R sqrt(A x) {
    //  return (R)((A)x)/(2*((R)std::sqrt((A)x)));
    //}
    float sqrt(float x) {
      //return (float)((float)x)/(2.F*((float)std::sqrt((float)x)));
      return x/(2.F*std::sqrt(x));
    }
    double sqrt(double x) {
      //return (double)((double)x)/(2.*((double)std::sqrt((double)x)));
      return x/(2.*std::sqrt(x));
    }
    long double sqrt(long double x) {
      //return (long double)((long double)x)/(2.L*((long double)std::sqrt((long double)x)));
      return x/(2.L*std::sqrt(x));
    }
    float sqrtf(float x) {
      //return (float)((float)x)/(2*((float)std::sqrtf((float)x)));
      return x/(2.F*std::sqrtf(x));
    }
  }
} // end namespace builtin_derivatives

#endif //CLAD_BUILTIN_DERIVATIVES
