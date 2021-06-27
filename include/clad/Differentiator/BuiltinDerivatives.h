//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_BUILTIN_DERIVATIVES
#define CLAD_BUILTIN_DERIVATIVES

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__ 
#else
#define CUDA_HOST_DEVICE
#endif

// Avoid assertion custom_derivative namespace not found. FIXME: This in future
// should go.
namespace custom_derivatives{}

#include <math.h>

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
  CUDA_HOST_DEVICE T abs_darg0(T x) {
    if (x >= 0)
      return 1;
    else
      return -1;
  }

  template<typename T>
  CUDA_HOST_DEVICE T exp_darg0(T x) {
    return exp(x);
  }

  template<typename T>
  CUDA_HOST_DEVICE T sin_darg0(T x) {
    return cos(x);
  }

  template<typename T>
  CUDA_HOST_DEVICE T cos_darg0(T x) {
    return (-1) * sin(x);
  }

  template<typename T>
  CUDA_HOST_DEVICE T sqrt_darg0(T x) {
    return ((T)1)/(((T)2)*sqrt(x));
  }

#ifdef MACOS
  float sqrtf_darg0(float x) {
    return 1.F/(2.F*sqrtf(x));
  }
#endif

  template<typename T1, typename T2>
  CUDA_HOST_DEVICE decltype(pow(T1(), T2())) pow_darg0(T1 x, T2 exponent) {
    return exponent * pow(x, exponent-((T2)1));
  }

  template <typename T1, typename T2>
  CUDA_HOST_DEVICE decltype(pow(T1(), T2())) pow_darg1(T1 x, T2 exponent) {
    return pow(x, exponent) * log(x);
  }

  template <typename T1, typename T2>
  CUDA_HOST_DEVICE void pow_grad(T1 x,
                                 T2 exponent,
                                 decltype(pow(T1(), T2()))* _d_x,
                                 decltype(pow(T1(), T2()))* _d_y) {
    *_d_x += pow_darg0(x, exponent);
    *_d_y += pow_darg1(x, exponent);
  }

  template <typename T>
  CUDA_HOST_DEVICE T log_darg0(T x) {
    return 1.0/x;
  }

  // FIXME: These math functions depend on promote_2 just like pow:
  // atan2
  // fmod
  // copysign
  // fdim
  // fmax
  // fmin
  // hypot
  // nextafter
  // remainder
  // remquo

} // end namespace builtin_derivatives

#endif //CLAD_BUILTIN_DERIVATIVES
