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

#include "clad/Differentiator/ArrayRef.h"
#include "clad/Differentiator/CladConfig.h"

#include <cmath>
namespace clad {
namespace custom_derivatives {
namespace std {
template <typename T> CUDA_HOST_DEVICE T abs_pushforward(T x, T d_x) {
  if (x >= 0)
    return d_x;
  else
    return -d_x;
}

template <typename T> CUDA_HOST_DEVICE T exp_pushforward(T x, T d_x) {
  return ::std::exp(x) * d_x;
}

template <typename T>
CUDA_HOST_DEVICE T exp_pushforward_pushforward(T x, T d_x, T d_x0, T d_d_x) {
  return ::std::exp(x) * d_x0 * d_x + ::std::exp(x) * d_d_x;
}

template <typename T> CUDA_HOST_DEVICE T sin_pushforward(T x, T d_x) {
  return ::std::cos(x) * d_x;
}

template <typename T> CUDA_HOST_DEVICE T cos_pushforward(T x, T d_x) {
  return (-1) * ::std::sin(x) * d_x;
}

template <typename T>
CUDA_HOST_DEVICE T sin_pushforward_pushforward(T x, T d_x, T d_x0, T d_d_x) {
  return cos_pushforward(x, d_x0) * d_x + ::std::cos(x) * d_d_x;
}

template <typename T>
CUDA_HOST_DEVICE T cos_pushforward_pushforward(T x, T d_x, T d_x0, T d_d_x) {
  return (-1) * (sin_pushforward(x, d_x0) * d_x + ::std::sin(x) * d_d_x);
}

template <typename T> CUDA_HOST_DEVICE T sqrt_pushforward(T x, T d_x) {
  return (((T)1) / (((T)2) * ::std::sqrt(x))) * d_x;
}

#ifdef MACOS
float sqrtf_pushforward(float x, float d_x) {
  return (1.F / (2.F * sqrtf(x))) * d_x;
}

#endif

template <typename T1, typename T2>
CUDA_HOST_DEVICE decltype(::std::pow(T1(), T2()))
pow_pushforward(T1 x, T2 exponent, T1 d_x, T2 d_exponent) {
  return (exponent * ::std::pow(x, exponent - 1)) * d_x +
         (::std::pow(x, exponent) * ::std::log(x)) * d_exponent;
}

template <typename T> CUDA_HOST_DEVICE T log_pushforward(T x, T d_x) {
  return (1.0 / x) * d_x;
}
} // namespace std
// These are required because C variants of mathematical functions are
// defined in global namespace.
using std::abs_pushforward;
using std::cos_pushforward;
using std::cos_pushforward_pushforward;
using std::exp_pushforward;
using std::exp_pushforward_pushforward;
using std::log_pushforward;
using std::pow_pushforward;
using std::sin_pushforward;
using std::sin_pushforward_pushforward;
using std::sqrt_pushforward;
} // namespace custom_derivatives
} // namespace clad

// TODO: Forward mode custom derivatives defined in this namespace can 
// be removed once custom derivative utilises pullback functions and all
// gradient/pullback functions use corresponding pushforward function instead of
// the `_darg` custom derivatives defined here.
namespace custom_derivatives {
// Why do we need to define 'std' namespace here when we are including the math
// library? 
// define functions' derivatives from std
namespace std {
// There are 4 overloads:
// float       sin( float arg );
// double      sin( double arg );
// long double sin( long double arg );
// double      sin( Integral arg ); (since C++11)
template <typename R, typename A> R sin(A x) {
  return (R)1;
  // return (R)::std::cos((A)x);
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

  template<typename T1, typename T2>
  CUDA_HOST_DEVICE decltype(pow(T1(), T2())) pow_darg0(T1 x, T2 exponent) {
    return exponent * pow(x, exponent-((T2)1));
  }

  template <typename T1, typename T2>
  CUDA_HOST_DEVICE decltype(pow(T1(), T2())) pow_darg1(T1 x, T2 exponent) {
    return pow(x, exponent) * log(x);
  }

  template<typename T1, typename T2>
  CUDA_HOST_DEVICE decltype(pow(T1(), T2())) pow_darg0_darg0(T1 x, T2 exponent) {
    return exponent * pow(x, exponent-((T2)1));
  }

  template<typename T1, typename T2>
  CUDA_HOST_DEVICE decltype(pow(T1(), T2())) pow_darg0_darg1(T1 x, T2 exponent) {
    return exponent * pow_darg1(x, exponent-((T2)1)) + pow(x, exponent-((T2)1));
  }

  template <typename T1, typename T2>
  CUDA_HOST_DEVICE decltype(pow(T1(), T2())) pow_darg1_darg0(T1 x, T2 exponent) {
    return exponent * pow_darg1(x, exponent-((T2)1)) + pow(x, exponent-((T2)1));
  }

  template <typename T1, typename T2>
  CUDA_HOST_DEVICE decltype(pow(T1(), T2())) pow_darg1_darg1(T1 x, T2 exponent) {
    return pow(x, exponent) * log(x) * log(x);
  }

  template <typename T1, typename T2>
  CUDA_HOST_DEVICE void
  pow_grad(T1 x, T2 exponent,
           clad::array_ref<decltype(::std::pow(T1(), T2()))> _d_x,
           clad::array_ref<decltype(::std::pow(T1(), T2()))> _d_y) {
    *_d_x += clad::custom_derivatives::std::pow_pushforward(x, exponent, static_cast<T1>(1),
                                               static_cast<T2>(0));
    *_d_y += clad::custom_derivatives::std::pow_pushforward(x, exponent, static_cast<T1>(0),
                                               static_cast<T2>(1));
  }

  // Do we need to define these functions? Ideally, clad can generate these
  // functions if they are required. 
  template <typename T1, typename T2>
  CUDA_HOST_DEVICE void
  pow_darg0_grad(T1 x, T2 exponent, clad::array_ref<decltype(pow(T1(), T2()))> _d_x,
           clad::array_ref<decltype(pow(T1(), T2()))> _d_y) {
    *_d_x += pow_darg0_darg0(x, exponent);
    *_d_y += pow_darg0_darg1(x, exponent);
  }

  template <typename T1, typename T2>
  CUDA_HOST_DEVICE void
  pow_darg1_grad(T1 x, T2 exponent, clad::array_ref<decltype(pow(T1(), T2()))> _d_x,
                 clad::array_ref<decltype(pow(T1(), T2()))> _d_y) {
    *_d_x += pow_darg1_darg0(x, exponent);
    *_d_y += pow_darg1_darg1(x, exponent);
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
