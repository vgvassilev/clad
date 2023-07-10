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
#ifdef __CUDACC__
template <typename T>
ValueAndPushforward<cudaError_t, cudaError_t>
cudaMalloc_pushforward(T** devPtr, size_t sz, T** d_devPtr, size_t d_sz)
    __attribute__((host)) {
  return {cudaMalloc(devPtr, sz), cudaMalloc(d_devPtr, sz)};
}

ValueAndPushforward<cudaError_t, cudaError_t>
cudaMemcpy_pushforward(void* destPtr, void* srcPtr, size_t count,
                       cudaMemcpyKind kind, void* d_destPtr, void* d_srcPtr,
                       size_t d_count) __attribute__((host)) {
  return {cudaMemcpy(destPtr, srcPtr, count, kind),
          cudaMemcpy(d_destPtr, d_srcPtr, count, kind)};
}

ValueAndPushforward<int, int> cudaDeviceSynchronize_pushforward()
    __attribute__((host)) {
  return {cudaDeviceSynchronize(), 0};
}
#endif

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

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> floor_pushforward(T x, T /*d_x*/) {
  return {::std::floor(x), (T)0};
}

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> ceil_pushforward(T x, T /*d_x*/) {
  return {::std::ceil(x), (T)0};
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
  auto val = ::std::pow(x, exponent);
  auto derivative = (exponent * ::std::pow(x, exponent - 1)) * d_x;
  // Only add directional derivative of base^exp w.r.t exp if the directional
  // seed d_exponent is non-zero. This is required because if base is less than or
  // equal to 0, then log(base) is undefined, and therefore if user only requested
  // directional derivative of base^exp w.r.t base -- which is valid --, the result would
  // be undefined because as per C++ valid number + NaN * 0 = NaN.
  if (d_exponent)
    derivative += (::std::pow(x, exponent) * ::std::log(x)) * d_exponent;
  return {val, derivative};
}

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> log_pushforward(T x, T d_x) {
  return {::std::log(x), static_cast<T>((1.0 / x) * d_x)};
}

template <typename T1, typename T2, typename T3>
CUDA_HOST_DEVICE void pow_pullback(T1 x, T2 exponent, T3 d_y,
                                   clad::array_ref<decltype(T1())> d_x,
                                   clad::array_ref<decltype(T2())> d_exponent) {
  auto t = pow_pushforward(x, exponent, static_cast<T1>(1), static_cast<T2>(0));
  *d_x += t.pushforward * d_y;
  t = pow_pushforward(x, exponent, static_cast<T1>(0), static_cast<T2>(1));
  *d_exponent += t.pushforward * d_y;
}

template <typename T1, typename T2, typename T3>
CUDA_HOST_DEVICE ValueAndPushforward<decltype(::std::fma(T1(), T2(), T3())),
                                     decltype(::std::fma(T1(), T2(), T3()))>
fma_pushforward(T1 a, T2 b, T3 c, T1 d_a, T2 d_b, T3 d_c) {
  auto val = ::std::fma(a, b, c);
  auto derivative = d_a * b + a * d_b + d_c;
  return {val, derivative};
}

template <typename T1, typename T2, typename T3, typename T4>
CUDA_HOST_DEVICE void fma_pullback(T1 a, T2 b, T3 c, T4 d_y,
                                   clad::array_ref<decltype(T1())> d_a,
                                   clad::array_ref<decltype(T2())> d_b,
                                   clad::array_ref<decltype(T3())> d_c) {
  *d_a += b * d_y;
  *d_b += a * d_y;
  *d_c += d_y;
}

} // namespace std
// These are required because C variants of mathematical functions are
// defined in global namespace.
using std::abs_pushforward;
using std::ceil_pushforward;
using std::cos_pushforward;
using std::exp_pushforward;
using std::floor_pushforward;
using std::fma_pullback;
using std::fma_pushforward;
using std::log_pushforward;
using std::pow_pullback;
using std::pow_pushforward;
using std::sin_pushforward;
using std::sqrt_pushforward;
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
