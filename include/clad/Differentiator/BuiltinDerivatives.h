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
template <typename T, typename U> struct ValueAndPushforward {
  T value;
  U pushforward;
};
namespace custom_derivatives {
namespace std {
template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> abs_pushforward(T x, T d_x) {
  if (x >= 0)
    return {x, d_x};
  else
    return {-x, -d_x};
}

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> exp_pushforward(T x, T d_x) {
  return {::std::exp(x), ::std::exp(x) * d_x};
}

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> sin_pushforward(T x, T d_x) {
  return {::std::sin(x), ::std::cos(x) * d_x};
}

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> cos_pushforward(T x, T d_x) {
  return {::std::cos(x), (-1) * ::std::sin(x) * d_x};
}

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> sqrt_pushforward(T x, T d_x) {
  return {::std::sqrt(x), (((T)1) / (((T)2) * ::std::sqrt(x))) * d_x};
}

#ifdef MACOS
ValueAndPushforward<float, float> sqrtf_pushforward(float x, float d_x) {
  return {sqrtf(x), (1.F / (2.F * sqrtf(x))) * d_x};
}

#endif

template <typename T1, typename T2>
CUDA_HOST_DEVICE ValueAndPushforward<decltype(::std::pow(T1(), T2())),
                                     decltype(::std::pow(T1(), T2()))>
pow_pushforward(T1 x, T2 exponent, T1 d_x, T2 d_exponent) {
  return {::std::pow(x, exponent),
          (exponent * ::std::pow(x, exponent - 1)) * d_x +
              (::std::pow(x, exponent) * ::std::log(x)) * d_exponent};
}

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> log_pushforward(T x, T d_x) {
  return {::std::log(x), static_cast<T>((1.0 / x) * d_x)};
}

template <typename T1, typename T2>
CUDA_HOST_DEVICE void
pow_pullback(T1 x, T2 exponent, decltype(::std::pow(T1(), T2())) d_y,
             clad::array_ref<decltype(::std::pow(T1(), T2()))> d_x,
             clad::array_ref<decltype(::std::pow(T1(), T2()))> d_exponent) {
  auto t = pow_pushforward(x, exponent, static_cast<T1>(1), static_cast<T2>(0));
  *d_x += t.pushforward * d_y;
  t = pow_pushforward(x, exponent, static_cast<T1>(0), static_cast<T2>(1));
  *d_exponent += t.pushforward * d_y;
}
} // namespace std
// These are required because C variants of mathematical functions are
// defined in global namespace.
using std::abs_pushforward;
using std::cos_pushforward;
using std::exp_pushforward;
using std::log_pushforward;
using std::pow_pushforward;
using std::sin_pushforward;
using std::sqrt_pushforward;
using std::pow_pullback;
} // namespace custom_derivatives
} // namespace clad

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
#endif //CLAD_BUILTIN_DERIVATIVES
