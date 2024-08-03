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

#include <algorithm>
#include <cmath>

namespace clad {
template <typename T, typename U> struct ValueAndPushforward {
  T value;
  U pushforward;

  // Define the cast operator from ValueAndPushforward<T, U> to
  // ValueAndPushforward<V, w> where V is convertible to T and W is
  // convertible to U.
  template <typename V = T, typename W = U>
  operator ValueAndPushforward<V, W>() const {
    return {static_cast<V>(value), static_cast<W>(pushforward)};
  }
};

/// It is used to identify constructor custom pushforwards. For
/// constructor custom pushforward functions, we cannot use the same
/// strategy which we use for custom pushforward for member
/// functions. Member functions custom pushforward have the following
/// signature:
///
/// mem_fn_pushforward(ClassName *c, ..., ClassName *d_c, ...)
///
/// We use the first argument 'ClassName *c' to determine the class of member
/// function for which the pushforward is defined.
///
/// In the case of constructor pushforward, there are no objects of the class
/// type passed to the constructor. Therefore, we cannot simply use arguments
/// to determine the class. To solve this, 'ConstructorPushforwardTag<T>' is
/// used. A custom_derivative pushforward for constructor is required to have
/// 'ConstructorPushforwardTag<T>' as the first argument, where 'T' is the
/// class for which constructor pushforward is defined.
template <class T> class ConstructorPushforwardTag {};

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

CUDA_HOST_DEVICE inline ValueAndPushforward<float, float>
__builtin_logf_pushforward(float x, float d_x) {
  return {__builtin_logf(x), (1.F / x) * d_x};
}

CUDA_HOST_DEVICE inline ValueAndPushforward<double, double>
__builtin_log_pushforward(double x, double d_x) {
  return {__builtin_log(x), (1.0 / x) * d_x};
}

CUDA_HOST_DEVICE inline ValueAndPushforward<double, double>
__builtin_pow_pushforward(double x, double exponent, double d_x,
                          double d_exponent) {
  auto val = __builtin_pow(x, exponent);
  double derivative = (exponent * __builtin_pow(x, exponent - 1)) * d_x;
  // Only add directional derivative of base^exp w.r.t exp if the directional
  // seed d_exponent is non-zero. This is required because if base is less than
  // or equal to 0, then log(base) is undefined, and therefore if user only
  // requested directional derivative of base^exp w.r.t base -- which is valid
  // --, the result would be undefined because as per C++ valid number + NaN * 0
  // = NaN.
  if (d_exponent)
    derivative += (__builtin_pow(x, exponent) * __builtin_log(x)) * d_exponent;
  return {val, derivative};
}

CUDA_HOST_DEVICE inline ValueAndPushforward<float, float>
__builtin_powf_pushforward(float x, float exponent, float d_x,
                           float d_exponent) {
  auto val = __builtin_powf(x, exponent);
  float derivative = (exponent * __builtin_powf(x, exponent - 1)) * d_x;
  // Only add directional derivative of base^exp w.r.t exp if the directional
  // seed d_exponent is non-zero. This is required because if base is less than
  // or equal to 0, then log(base) is undefined, and therefore if user only
  // requested directional derivative of base^exp w.r.t base -- which is valid
  // --, the result would be undefined because as per C++ valid number + NaN * 0
  // = NaN.
  if (d_exponent)
    derivative +=
        (__builtin_powf(x, exponent) * __builtin_logf(x)) * d_exponent;
  return {val, derivative};
}

CUDA_HOST_DEVICE inline void __builtin_pow_pullback(double x, double exponent,
                                                    double d_y, double* d_x,
                                                    double* d_exponent) {
  auto t =
      __builtin_pow_pushforward(x, exponent, /*d_x=*/1., /*d_exponent=*/0.);
  *d_x += t.pushforward * d_y;
  t = __builtin_pow_pushforward(x, exponent, /*d_x=*/0., /*d_exponent=*/1.);
  *d_exponent += t.pushforward * d_y;
}

CUDA_HOST_DEVICE inline void __builtin_powf_pullback(float x, float exponent,
                                                     float d_y, float* d_x,
                                                     float* d_exponent) {
  auto t =
      __builtin_powf_pushforward(x, exponent, /*d_x=*/1., /*d_exponent=*/0.);
  *d_x += t.pushforward * d_y;
  t = __builtin_powf_pushforward(x, exponent, /*d_x=*/0., /*d_exponent=*/1.);
  *d_exponent += t.pushforward * d_y;
}

// FIXME: Add the rest of the __builtin_ routines for log, sqrt, abs, etc.

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
CUDA_HOST_DEVICE void pow_pullback(T1 x, T2 exponent, T3 d_y, T1* d_x,
                                   T2* d_exponent) {
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
CUDA_HOST_DEVICE void fma_pullback(T1 a, T2 b, T3 c, T4 d_y, T1* d_a, T2* d_b,
                                   T3* d_c) {
  *d_a += b * d_y;
  *d_b += a * d_y;
  *d_c += d_y;
}

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T>
min_pushforward(const T& a, const T& b, const T& d_a, const T& d_b) {
  return {::std::min(a, b), a < b ? d_a : d_b};
}

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T>
max_pushforward(const T& a, const T& b, const T& d_a, const T& d_b) {
  return {::std::max(a, b), a < b ? d_b : d_a};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void min_pullback(const T& a, const T& b, U d_y, T* d_a,
                                   T* d_b) {
  if (a < b)
    *d_a += d_y;
  else
    *d_b += d_y;
}

template <typename T, typename U>
CUDA_HOST_DEVICE void max_pullback(const T& a, const T& b, U d_y, T* d_a,
                                   T* d_b) {
  if (a < b)
    *d_b += d_y;
  else
    *d_a += d_y;
}

#if __cplusplus >= 201703L
template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T>
clamp_pushforward(const T& v, const T& lo, const T& hi, const T& d_v,
                  const T& d_lo, const T& d_hi) {
  return {::std::clamp(v, lo, hi), v < lo ? d_lo : hi < v ? d_hi : d_v};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void clamp_pullback(const T& v, const T& lo, const T& hi,
                                     const U& d_y, T* d_v, T* d_lo, T* d_hi) {
  if (v < lo)
    *d_lo += d_y;
  else if (hi < v)
    *d_hi += d_y;
  else
    *d_v += d_y;
}
#endif

} // namespace std

// NOLINTBEGIN(cppcoreguidelines-no-malloc)
// NOLINTBEGIN(cppcoreguidelines-owning-memory)
inline ValueAndPushforward<void*, void*> malloc_pushforward(size_t sz,
                                                            size_t d_sz) {
  return {malloc(sz), malloc(sz)};
}

inline ValueAndPushforward<void*, void*>
calloc_pushforward(size_t n, size_t sz, size_t d_n, size_t d_sz) {
  return {calloc(n, sz), calloc(n, sz)};
}

inline ValueAndPushforward<void*, void*>
realloc_pushforward(void* ptr, size_t sz, void* d_ptr, size_t d_sz) {
  return {realloc(ptr, sz), realloc(d_ptr, sz)};
}

inline void free_pushforward(void* ptr, void* d_ptr) {
  free(ptr);
  free(d_ptr);
}
// NOLINTEND(cppcoreguidelines-owning-memory)
// NOLINTEND(cppcoreguidelines-no-malloc)

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
using std::max_pullback;
using std::max_pushforward;
using std::min_pullback;
using std::min_pushforward;
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
