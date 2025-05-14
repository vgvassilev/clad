//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_BUILTIN_DERIVATIVES
#define CLAD_BUILTIN_DERIVATIVES

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

template <typename T, typename U>
ValueAndPushforward<T, U> make_value_and_pushforward(T value, U pushforward) {
  return {value, pushforward};
}

template <typename T, typename U> struct ValueAndAdjoint {
  T value;
  U adjoint;
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

template <class T> class ConstructorReverseForwTag {};

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

template <typename T>
__global__ void atomicAdd_kernel(T* destPtr, T* srcPtr, size_t N) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    atomicAdd(&destPtr[i], srcPtr[i]);
}

template <typename T>
void cudaMemcpy_pullback(T* destPtr, const T* srcPtr, size_t count,
                         cudaMemcpyKind kind, cudaError_t d_y, T* d_destPtr,
                         T* d_srcPtr, size_t* d_count, cudaMemcpyKind* d_kind)
    __attribute__((host)) {
  T* aux_destPtr = nullptr;
  if (kind == cudaMemcpyDeviceToHost) {
    *d_kind = cudaMemcpyHostToDevice;
    cudaMalloc(&aux_destPtr, count);
  } else if (kind == cudaMemcpyHostToDevice) {
    *d_kind = cudaMemcpyDeviceToHost;
    aux_destPtr = (T*)malloc(count);
  }
  cudaDeviceSynchronize(); // needed in case user uses another stream for
                           // kernel execution besides the default one
  cudaMemcpy(aux_destPtr, d_destPtr, count, *d_kind);
  size_t N = count / sizeof(T);
  if (kind == cudaMemcpyDeviceToHost) {
    // d_kind is host to device, so d_srcPtr is a device pointer
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t maxThreads = deviceProp.maxThreadsPerBlock;
    size_t maxBlocks = deviceProp.maxGridSize[0];

    size_t numThreads = std::min(maxThreads, N);
    size_t numBlocks = std::min(maxBlocks, (N + numThreads - 1) / numThreads);
    custom_derivatives::atomicAdd_kernel<<<numBlocks, numThreads>>>(
        d_srcPtr, aux_destPtr, N);
    cudaDeviceSynchronize(); // needed in case the user uses another stream for
                             // kernel execution besides the default one, so we
                             // need to make sure the data are updated before
                             // continuing with the rest of the code
    cudaFree(aux_destPtr);
  } else if (kind == cudaMemcpyHostToDevice) {
    // d_kind is device to host, so d_srcPtr is a host pointer
    for (size_t i = 0; i < N; i++)
      d_srcPtr[i] += aux_destPtr[i];
    free(aux_destPtr);
  }
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
  if (exponent == 0 && d_exponent == 0)
    return {val, 0};
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
  if (exponent == 0 && d_exponent == 0)
    return {val, 0};
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

// 1. Basic Functions

// 1.1 abs, labs, llabs, imaxabs

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> abs_pushforward(T x, dT d_x) {
  if (x >= 0)
    return {x, d_x};
  else
    return {-x, -d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void abs_pullback(T x, U d_y, T* d_x) {
  if (x >= 0)
    *d_x += d_y;
  else
    *d_x -= d_y;
}

// pushforward for labs, llabs, imaxabs
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> labs_pushforward(T x, dT d_x) {
  return abs_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> llabs_pushforward(T x, dT d_x) {
  return abs_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> imaxabs_pushforward(T x, dT d_x) {
  return abs_pushforward(x, d_x);
}

// pullback for labs, llabs, imaxabs
template <typename T, typename U>
CUDA_HOST_DEVICE void labs_pullback(T x, U d_y, T* d_x) {
  abs_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void llabs_pullback(T x, U d_y, T* d_x) {
  abs_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void imaxabs_pullback(T x, U d_y, T* d_x) {
  abs_pullback(x, d_y, d_x);
}

// 1.2 fabs, fabsf, fabsl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fabs_pushforward(T x, dT d_x) {
  return abs_pushforward(x, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fabs_pullback(T x, U d_y, T* d_x) {
  abs_pullback(x, d_y, d_x);
}

// pushforward for fabsf, fasbsl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fabsf_pushforward(T x, dT d_x) {
  return fabs_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fabsl_pushforward(T x, dT d_x) {
  return fabs_pushforward(x, d_x);
}

// pullback for fabsf, fabsl
template <typename T, typename U>
CUDA_HOST_DEVICE void fabsf_pullback(T x, U d_y, T* d_x) {
  fabs_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fabsl_pullback(T x, U d_y, T* d_x) {
  fabs_pullback(x, d_y, d_x);
}

// 1.3 fmod, fmodf, fmodl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fmod_pushforward(T x, T y, dT d_x,
                                                             dT d_y) {
  return {::std::fmod(x, y), d_x - d_y * ::std::floor(x / y)};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fmod_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  *d_x += d_z;
  *d_y -= d_z * ::std::floor(x / y);
}

// pushforward for fmodf, fmodl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fmodf_pushforward(T x, T y, dT d_x,
                                                              dT d_y) {
  return fmod_pushforward(x, y, d_x, d_y);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fmodl_pushforward(T x, T y, dT d_x,
                                                              dT d_y) {
  return fmod_pushforward(x, y, d_x, d_y);
}

// pullback for fmodf, fmodl
template <typename T, typename U>
CUDA_HOST_DEVICE void fmodf_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  fmod_pullback(x, y, d_z, d_x, d_y);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fmodl_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  fmod_pullback(x, y, d_z, d_x, d_y);
}

// 1.4 remainder, remainderf, remainderl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT>
remainder_pushforward(T x, T y, dT d_x, dT d_y) {
  return {::std::remainder(x, y), d_x - d_y * ::std::floor(x / y)};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void remainder_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  *d_x += d_z;
  *d_y -= d_z * ::std::floor(x / y);
}

// pushforward for remainderf, remainderl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT>
remainderf_pushforward(T x, T y, dT d_x, dT d_y) {
  return remainder_pushforward(x, y, d_x, d_y);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT>
remainderl_pushforward(T x, T y, dT d_x, dT d_y) {
  return remainder_pushforward(x, y, d_x, d_y);
}

// pullback for remainderf, remainderl
template <typename T, typename U>
CUDA_HOST_DEVICE void remainderf_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  remainder_pullback(x, y, d_z, d_x, d_y);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void remainderl_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  remainder_pullback(x, y, d_z, d_x, d_y);
}

// 1.5 remquo, remquof, remquol
// To be implemented

// 1.6 fma, fmaf, fmal
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

// pushforward for fmaf, fmal
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT>
fmaf_pushforward(T x, T y, T z, dT d_x, dT d_y, dT d_z) {
  return fma_pushforward(x, y, z, d_x, d_y, d_z);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT>
fmal_pushforward(T x, T y, T z, dT d_x, dT d_y, dT d_z) {
  return fma_pushforward(x, y, z, d_x, d_y, d_z);
}

// pullback for fmaf, fmal
template <typename T, typename U>
CUDA_HOST_DEVICE void fmaf_pullback(T x, T y, T z, U d_w, T* d_x, T* d_y,
                                    T* d_z) {
  fma_pullback(x, y, z, d_w, d_x, d_y, d_z);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fmal_pullback(T x, T y, T z, U d_w, T* d_x, T* d_y,
                                    T* d_z) {
  fma_pullback(x, y, z, d_w, d_x, d_y, d_z);
}

// 1.7 fmax, fmaxf, fmaxl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fmax_pushforward(T x, T y, dT d_x,
                                                             dT d_y) {
  return {::std::fmax(x, y), (x > y) ? d_x : d_y};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fmax_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  if (x > y)
    *d_x += d_z;
  else
    *d_y += d_z;
}

// pushforward for fmaxf, fmaxl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fmaxf_pushforward(T x, T y, dT d_x,
                                                              dT d_y) {
  return fmax_pushforward(x, y, d_x, d_y);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fmaxl_pushforward(T x, T y, dT d_x,
                                                              dT d_y) {
  return fmax_pushforward(x, y, d_x, d_y);
}

// pullback for fmaxf, fmaxl
template <typename T, typename U>
CUDA_HOST_DEVICE void fmaxf_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  fmax_pullback(x, y, d_z, d_x, d_y);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fmaxl_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  fmax_pullback(x, y, d_z, d_x, d_y);
}

// 1.8 fmin, fminf, fminl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fmin_pushforward(T x, T y, dT d_x,
                                                             dT d_y) {
  return {::std::fmin(x, y), (x < y) ? d_x : d_y};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fmin_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  if (x < y)
    *d_x += d_z;
  else
    *d_y += d_z;
}

// pushforward for fminf, fminl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fminf_pushforward(T x, T y, dT d_x,
                                                              dT d_y) {
  return fmin_pushforward(x, y, d_x, d_y);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fminl_pushforward(T x, T y, dT d_x,
                                                              dT d_y) {
  return fmin_pushforward(x, y, d_x, d_y);
}

// pullback for fminf, fminl
template <typename T, typename U>
CUDA_HOST_DEVICE void fminf_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  fmin_pullback(x, y, d_z, d_x, d_y);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fminl_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  fmin_pullback(x, y, d_z, d_x, d_y);
}

// 1.9 fdim, fdimf, fdiml
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fdim_pushforward(T x, T y, dT d_x,
                                                             dT d_y) {
  return {::std::fdim(x, y), (x > y) ? d_x : 0};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fdim_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  if (x > y)
    *d_x += d_z;
}

// pushforward for fdimf, fdiml
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fdimf_pushforward(T x, T y, dT d_x,
                                                              dT d_y) {
  return fdim_pushforward(x, y, d_x, d_y);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> fdiml_pushforward(T x, T y, dT d_x,
                                                              dT d_y) {
  return fdim_pushforward(x, y, d_x, d_y);
}

// pullback for fdimf, fdiml
template <typename T, typename U>
CUDA_HOST_DEVICE void fdimf_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  fdim_pullback(x, y, d_z, d_x, d_y);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void fdiml_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  fdim_pullback(x, y, d_z, d_x, d_y);
}

// 2. Exponential Functions

// 2.1 exp, expf, expl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> exp_pushforward(T x, dT d_x) {
  return {::std::exp(x), ::std::exp(x) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void exp_pullback(T x, U d_y, T* d_x) {
  *d_x += d_y * ::std::exp(x);
}

// pushforward for expf, expl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> expf_pushforward(T x, dT d_x) {
  return exp_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> expl_pushforward(T x, dT d_x) {
  return exp_pushforward(x, d_x);
}

// pullback for expf, expl
template <typename T, typename U>
CUDA_HOST_DEVICE void expf_pullback(T x, U d_y, T* d_x) {
  exp_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void expl_pullback(T x, U d_y, T* d_x) {
  exp_pullback(x, d_y, d_x);
}

// 2.2 exp2, exp2f, exp2l
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> exp2_pushforward(T x, dT d_x) {
  return {::std::exp2(x), ::std::exp2(x) * ::std::log(2) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void exp2_pullback(T x, U d_y, T* d_x) {
  *d_x += d_y * ::std::log(2) * ::std::exp2(x);
}

// pushforward for exp2f, exp2l
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> exp2f_pushforward(T x, dT d_x) {
  return exp2_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> exp2l_pushforward(T x, dT d_x) {
  return exp2_pushforward(x, d_x);
}

// pullback for exp2f, exp2l
template <typename T, typename U>
CUDA_HOST_DEVICE void exp2f_pullback(T x, U d_y, T* d_x) {
  exp2_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void exp2l_pullback(T x, U d_y, T* d_x) {
  exp2_pullback(x, d_y, d_x);
}

// 2.3 expm1, expm1f, expm1l
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> expm1_pushforward(T x, dT d_x) {
  return {::std::expm1(x), ::std::exp(x) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void expm1_pullback(T x, U d_y, T* d_x) {
  *d_x += d_y * ::std::exp(x);
}

// pushforward for expm1f, expm1l
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> expm1f_pushforward(T x, dT d_x) {
  return expm1_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> expm1l_pushforward(T x, dT d_x) {
  return expm1_pushforward(x, d_x);
}

// pullback for expm1f, expm1l
template <typename T, typename U>
CUDA_HOST_DEVICE void expm1f_pullback(T x, U d_y, T* d_x) {
  expm1_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void expm1l_pullback(T x, U d_y, T* d_x) {
  expm1_pullback(x, d_y, d_x);
}

// 3. Logarithmic Functions

// 3.1 log, logf, logl
template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> log_pushforward(T x, T d_x) {
  return {::std::log(x), static_cast<T>((1.0 / x) * d_x)};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void log_pullback(T x, U d_y, T* d_x) {
  *d_x += d_y * (1.0 / x);
}

// pushforward for logf, logl
template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> logf_pushforward(T x, T d_x) {
  return {log_pushforward(x, d_x)};
}

template <typename T>
CUDA_HOST_DEVICE ValueAndPushforward<T, T> logl_pushforward(T x, T d_x) {
  return {log_pushforward(x, d_x)};
}

// pullback for logf, logl
template <typename T, typename U>
CUDA_HOST_DEVICE void logf_pullback(T x, U d_y, T* d_x) {
  log_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void logl_pullback(T x, U d_y, T* d_x) {
  log_pullback(x, d_y, d_x);
}

// 3.2 log10, log10f, log10l
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> log10_pushforward(T x, dT d_x) {
  return {::std::log10(x), (1.0 / (x * ::std::log(10))) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void log10_pullback(T x, U d_y, T* d_x) {
  *d_x += d_y * (1.0 / (x * ::std::log(10)));
}

// pushforward for log10f, log10l
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> log10f_pushforward(T x, dT d_x) {
  return log10_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> log10l_pushforward(T x, dT d_x) {
  return log10_pushforward(x, d_x);
}

// pullback for log10f, log10l
template <typename T, typename U>
CUDA_HOST_DEVICE void log10f_pullback(T x, U d_y, T* d_x) {
  log10_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void log10l_pullback(T x, U d_y, T* d_x) {
  log10_pullback(x, d_y, d_x);
}

// 3.3 log2, log2f, log2l
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> log2_pushforward(T x, dT d_x) {
  return {::std::log2(x), (1.0 / (x * ::std::log(2))) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void log2_pullback(T x, U d_y, T* d_x) {
  *d_x += d_y * (1.0 / (x * ::std::log(2)));
}

// pushforward for log2f, log2l
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> log2f_pushforward(T x, dT d_x) {
  return log2_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> log2l_pushforward(T x, dT d_x) {
  return log2_pushforward(x, d_x);
}

// pullback for log2f, log2l
template <typename T, typename U>
CUDA_HOST_DEVICE void log2f_pullback(T x, U d_y, T* d_x) {
  log2_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void log2l_pullback(T x, U d_y, T* d_x) {
  log2_pullback(x, d_y, d_x);
}

// 3.4 log1p, log1pf, log1pl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> log1p_pushforward(T x, dT d_x) {
  return {::std::log1p(x), (1.0 / (1 + x)) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void log1p_pullback(T x, U d_y, T* d_x) {
  *d_x += d_y * (1.0 / (1 + x));
}

// pushforward for log1pf, log1pl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> log1pf_pushforward(T x, dT d_x) {
  return log1p_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> log1pl_pushforward(T x, dT d_x) {
  return log1p_pushforward(x, d_x);
}
// pullback for log1pf, log1pl
template <typename T, typename U>
CUDA_HOST_DEVICE void log1pf_pullback(T x, U d_y, T* d_x) {
  log1p_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void log1pl_pullback(T x, U d_y, T* d_x) {
  log1p_pullback(x, d_y, d_x);
}

// 4. Trigonometric Functions
// 4.1 sin, sinf, sinl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> sin_pushforward(T x, dT d_x) {
  return {::std::sin(x), ::std::cos(x) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void sin_pullback(T x, U d_y, T* d_x) {
  *d_x += d_y * ::std::cos(x);
}

// pushforward for sinf, sinl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> sinf_pushforward(T x, dT d_x) {
  return sin_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> sinl_pushforward(T x, dT d_x) {
  return sin_pushforward(x, d_x);
}

// pullback for sinf, sinl
template <typename T, typename U>
CUDA_HOST_DEVICE void sinf_pullback(T x, U d_y, T* d_x) {
  sin_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void sinl_pullback(T x, U d_y, T* d_x) {
  sin_pullback(x, d_y, d_x);
}

// 4.2 cos, cosf, cosl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> cos_pushforward(T x, dT d_x) {
  return {::std::cos(x), (-1) * ::std::sin(x) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void cos_pullback(T x, U d_y, T* d_x) {
  *d_x += d_y * (-1) * ::std::sin(x);
}

// pushforward for cosf, cosl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> cosf_pushforward(T x, dT d_x) {
  return cos_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> cosl_pushforward(T x, dT d_x) {
  return cos_pushforward(x, d_x);
}

// pullback for cosf, cosl
template <typename T, typename U>
CUDA_HOST_DEVICE void cosf_pullback(T x, U d_y, T* d_x) {
  cos_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void cosl_pullback(T x, U d_y, T* d_x) {
  cos_pullback(x, d_y, d_x);
}

// 4.3 tan, tanf, tanl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> tan_pushforward(T x, dT d_x) {
  T tanx = ::std::tan(x);
  return {tanx, (1 + tanx * tanx) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void tan_pullback(T x, U d_y, T* d_x) {
  T tanx = ::std::tan(x);
  *d_x += d_y * (1 + tanx * tanx);
}

// pushforward for tanf, tanl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> tanf_pushforward(T x, dT d_x) {
  return tan_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> tanl_pushforward(T x, dT d_x) {
  return tan_pushforward(x, d_x);
}

// pullback for tanf, tanl
template <typename T, typename U>
CUDA_HOST_DEVICE void tanf_pullback(T x, U d_y, T* d_x) {
  tan_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void tanl_pullback(T x, U d_y, T* d_x) {
  tan_pullback(x, d_y, d_x);
}

// 4.4 asin, asinf, asinl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> asin_pushforward(T x, dT d_x) {
  return {::std::asin(x), d_x / ::std::sqrt(1 - x * x)};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void asin_pullback(T x, U d_z, T* d_x) {
  *d_x += d_z / ::std::sqrt(1 - x * x);
}

// pushforward for asinf, asinl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> asinf_pushforward(T x, dT d_x) {
  return asin_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> asinl_pushforward(T x, dT d_x) {
  return asin_pushforward(x, d_x);
}

// pullback for asinf, asinl
template <typename T, typename U>
CUDA_HOST_DEVICE void asinf_pullback(T x, U d_z, T* d_x) {
  asin_pullback(x, d_z, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void asinl_pullback(T x, U d_z, T* d_x) {
  asin_pullback(x, d_z, d_x);
}

// 4.5 acos, acosf, acosl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> acos_pushforward(T x, dT d_x) {
  return {::std::acos(x), ((-1) / (::std::sqrt(1 - x * x))) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void acos_pullback(T x, U d_z, T* d_x) {
  *d_x += d_z * ((-1) / (::std::sqrt(1 - x * x)));
}

// pushforward for acosf, acosl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> acosf_pushforward(T x, dT d_x) {
  return acos_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> acosl_pushforward(T x, dT d_x) {
  return acos_pushforward(x, d_x);
}

// pullback for acosf, acosl
template <typename T, typename U>
CUDA_HOST_DEVICE void acosf_pullback(T x, U d_z, T* d_x) {
  acos_pullback(x, d_z, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void acosl_pullback(T x, U d_z, T* d_x) {
  acos_pullback(x, d_z, d_x);
}

// 4.6 atan, atanf, atanl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> atan_pushforward(T x, dT d_x) {
  return {::std::atan(x), d_x / (1 + x * x)};
}

template <typename T, typename dT>
CUDA_HOST_DEVICE void atan_pullback(T x, T d_y, T* d_x) {
  *d_x += d_y / (1 + x * x);
}

// pushforward for atanf, atanl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> atanf_pushforward(T x, dT d_x) {
  return atan_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> atanl_pushforward(T x, dT d_x) {
  return atan_pushforward(x, d_x);
}

// pullback for atanf, atanl
template <typename T, typename U>
CUDA_HOST_DEVICE void atanf_pullback(T x, U d_y, T* d_x) {
  atan_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void atanl_pullback(T x, U d_y, T* d_x) {
  atan_pullback(x, d_y, d_x);
}

// 4.7 atan2, atan2f, atan2l
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> atan2_pushforward(T y, T x, dT d_y,
                                                              dT d_x) {
  return {::std::atan2(y, x),
          -(y / ((x * x) + (y * y))) * d_x + x / ((x * x) + (y * y)) * d_y};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void atan2_pullback(T y, T x, U d_z, T* d_y, T* d_x) {
  *d_y += x / ((x * x) + (y * y)) * d_z;

  *d_x += -(y / ((x * x) + (y * y))) * d_z;
}

// pushforward for atan2f, atan2l
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> atan2f_pushforward(T y, T x, dT d_y,
                                                               dT d_x) {
  return atan2_pushforward(y, x, d_y, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> atan2l_pushforward(T y, T x, dT d_y,
                                                               dT d_x) {
  return atan2_pushforward(y, x, d_y, d_x);
}

// pullback for atan2f, atan2l
template <typename T, typename U>
CUDA_HOST_DEVICE void atan2f_pullback(T y, T x, U d_z, T* d_y, T* d_x) {
  atan2_pullback(y, x, d_z, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void atan2l_pullback(T y, T x, U d_z, T* d_y, T* d_x) {
  atan2_pullback(y, x, d_z, d_y, d_x);
}

// 5. Hyperbolic Functions
// 5.1 sinh, sinhf, sinhl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> sinh_pushforward(T x, dT d_x) {
  return {::std::sinh(x), ::std::cosh(x) * d_x};
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> sinh_pullback(T x, T d_y, T* d_x) {
  *d_x += ::std::cosh(x) * d_y;
}

// pushforward for sinhf, sinhl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> sinhf_pushforward(T x, dT d_x) {
  return sinh_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> sinhl_pushforward(T x, dT d_x) {
  return sinh_pushforward(x, d_x);
}

// pullback for sinhf, sinhl
template <typename T, typename U>
CUDA_HOST_DEVICE void sinhf_pullback(T x, U d_y, T* d_x) {
  sinh_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void sinhl_pullback(T x, U d_y, T* d_x) {
  sinh_pullback(x, d_y, d_x);
}

// 5.2 cosh, coshf, coshl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> cosh_pushforward(T x, dT d_x) {
  return {::std::cosh(x), ::std::sinh(x) * d_x};
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> cosh_pullback(T x, T d_y, T* d_x) {
  *d_x += ::std::sinh(x) * d_y;
}

// pushforward for coshf, coshl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> coshf_pushforward(T x, dT d_x) {
  return cosh_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> coshl_pushforward(T x, dT d_x) {
  return cosh_pushforward(x, d_x);
}

// pullback for coshf, coshl
template <typename T, typename U>
CUDA_HOST_DEVICE void coshf_pullback(T x, U d_y, T* d_x) {
  cosh_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void coshl_pullback(T x, U d_y, T* d_x) {
  cosh_pullback(x, d_y, d_x);
}

// 5.3 tanh, tanhf, tanhl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> tanh_pushforward(T x, dT d_x) {
  T tanhx = ::std::tanh(x);
  return {tanhx, (1 - tanhx * tanhx) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void tanh_pullback(T x, U d_y, T* d_x) {
  T tanhx = ::std::tanh(x);
  *d_x += d_y * (1 - tanhx * tanhx);
}

// pushforward for tanhf, tanhl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> tanhf_pushforward(T x, dT d_x) {
  return tanh_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> tanhl_pushforward(T x, dT d_x) {
  return tanh_pushforward(x, d_x);
}

// pullback for tanhf, tanhl
template <typename T, typename U>
CUDA_HOST_DEVICE void tanhf_pullback(T x, U d_y, T* d_x) {
  tanh_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void tanhl_pullback(T x, U d_y, T* d_x) {
  tanh_pullback(x, d_y, d_x);
}

// 5.4 asinh, asinhf, asinhl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> asinh_pushforward(T x, dT d_x) {
  return {::std::asinh(x), d_x / ::std::sqrt(1 + x * x)};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void asinh_pullback(T x, U d_z, T* d_x) {
  *d_x += d_z / ::std::sqrt(1 + x * x);
}

// pushforward for asinhf, asinhl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> asinhf_pushforward(T x, dT d_x) {
  return asinh_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> asinhl_pushforward(T x, dT d_x) {
  return asinh_pushforward(x, d_x);
}

// pullback for asinhf, asinhl
template <typename T, typename U>
CUDA_HOST_DEVICE void asinhf_pullback(T x, U d_z, T* d_x) {
  asinh_pullback(x, d_z, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void asinhl_pullback(T x, U d_z, T* d_x) {
  asinh_pullback(x, d_z, d_x);
}

// 5.5 acosh, acoshf, acoshl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> acosh_pushforward(T x, dT d_x) {
  return {::std::acosh(x), d_x / ::std::sqrt(x * x - 1)};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void acosh_pullback(T x, U d_z, T* d_x) {
  *d_x += d_z / ::std::sqrt(x * x - 1);
}

// pushforward for acoshf, acoshl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> acoshf_pushforward(T x, dT d_x) {
  return acosh_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> acoshl_pushforward(T x, dT d_x) {
  return acosh_pushforward(x, d_x);
}

// pullback for acoshf, acoshl
template <typename T, typename U>
CUDA_HOST_DEVICE void acoshf_pullback(T x, U d_z, T* d_x) {
  acosh_pullback(x, d_z, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void acoshl_pullback(T x, U d_z, T* d_x) {
  acosh_pullback(x, d_z, d_x);
}

// 5.6 atanh, atanhf, atanhl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> atanh_pushforward(T x, dT d_x) {
  return {::std::atanh(x), d_x / (1 - x * x)};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void atanh_pullback(T x, U d_z, T* d_x) {
  *d_x += d_z / (1 - x * x);
}

// pushforward for atanhf, atanhl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> atanhf_pushforward(T x, dT d_x) {
  return atanh_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> atanhl_pushforward(T x, dT d_x) {
  return atanh_pushforward(x, d_x);
}

// pullback for atanhf, atanhl
template <typename T, typename U>
CUDA_HOST_DEVICE void atanhf_pullback(T x, U d_z, T* d_x) {
  atanh_pullback(x, d_z, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void atanhl_pullback(T x, U d_z, T* d_x) {
  atanh_pullback(x, d_z, d_x);
}

// 6. Error Functions:
// 6.1 erf, erff, erfl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> erf_pushforward(T x, dT d_x) {
  return {::std::erf(x), (2 / ::std::sqrt(M_PI)) * ::std::exp(-x * x) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void erf_pullback(T x, U d_y, T* d_x) {
  *d_x += (2 / ::std::sqrt(M_PI)) * ::std::exp(-x * x) * d_y;
}

// pushforward for erff, erfl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> erff_pushforward(T x, dT d_x) {
  return erf_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> erfl_pushforward(T x, dT d_x) {
  return erf_pushforward(x, d_x);
}

// pullback for erff, erfl
template <typename T, typename U>
CUDA_HOST_DEVICE void erff_pullback(T x, U d_y, T* d_x) {
  erf_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void erfl_pullback(T x, U d_y, T* d_x) {
  erf_pullback(x, d_y, d_x);
}

// 6.2 erfc, erfcf, erfcl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> erfc_pushforward(T x, dT d_x) {
  return {::std::erfc(x), (-2 / ::std::sqrt(M_PI)) * ::std::exp(-x * x) * d_x};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void erfc_pullback(T x, U d_y, T* d_x) {
  *d_x += (-2 / ::std::sqrt(M_PI)) * ::std::exp(-x * x) * d_y;
}

// pushforward for erfcf, erfcl
template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> erfcf_pushforward(T x, dT d_x) {
  return erfc_pushforward(x, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> erfcl_pushforward(T x, dT d_x) {
  return erfc_pushforward(x, d_x);
}

// pullback for erfcf, erfcl
template <typename T, typename U>
CUDA_HOST_DEVICE void erfcf_pullback(T x, U d_y, T* d_x) {
  erfc_pullback(x, d_y, d_x);
}

template <typename T, typename U>
CUDA_HOST_DEVICE void erfcl_pullback(T x, U d_y, T* d_x) {
  erfc_pullback(x, d_y, d_x);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> sqrt_pushforward(T x, dT d_x) {
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

template <typename T, typename dT> struct AdjOutType {
  using type = T;
};

template <typename T, typename dT> struct AdjOutType<T, clad::array<dT>> {
  using type = clad::array<T>;
};

template <typename T1, typename T2, typename dT1, typename dT2,
          typename T_out = decltype(::std::pow(T1(), T2())),
          typename dT_out = typename AdjOutType<T_out, dT1>::type>
CUDA_HOST_DEVICE ValueAndPushforward<T_out, dT_out>
pow_pushforward(T1 x, T2 exponent, dT1 d_x, dT2 d_exponent) {
  T_out val = ::std::pow(x, exponent);
  if (exponent == static_cast<T2>(0) && d_exponent == static_cast<dT2>(0))
    return {val, static_cast<dT_out>(0)};
  dT_out derivative = (exponent * ::std::pow(x, exponent - 1)) * d_x;
  // Only add directional derivative of base^exp w.r.t exp if the directional
  // seed d_exponent is non-zero. This is required because if base is less than
  // or equal to 0, then log(base) is undefined, and therefore if user only
  // requested directional derivative of base^exp w.r.t base -- which is valid
  // --, the result would be undefined because as per C++ valid number + NaN * 0
  // = NaN.
  if (d_exponent)
    derivative += (::std::pow(x, exponent) * ::std::log(x)) * d_exponent;
  return {val, derivative};
}

template <typename T1, typename T2, typename T3>
CUDA_HOST_DEVICE void pow_pullback(T1 x, T2 exponent, T3 d_y, T1* d_x,
                                   T2* d_exponent) {
  auto t = pow_pushforward(x, exponent, static_cast<T1>(1), static_cast<T2>(0));
  *d_x += t.pushforward * d_y;
  t = pow_pushforward(x, exponent, static_cast<T1>(0), static_cast<T2>(1));
  *d_exponent += t.pushforward * d_y;
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
                                     U d_y, T* d_v, T* d_lo, T* d_hi) {
  if (v < lo)
    *d_lo += d_y;
  else if (hi < v)
    *d_hi += d_y;
  else
    *d_v += d_y;
}
#endif

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> cbrt_pushforward(T x, dT d_x) {
  T cbrtx = ::std::cbrt(x);
  return {cbrtx, d_x / (3 * cbrtx * cbrtx)};
}

template <typename T, typename dT>
CUDA_HOST_DEVICE void cbrt_pullback(T x, T d_y, T* d_x) {
  T cbrtx = ::std::cbrt(x);
  *d_x += d_y / (3 * cbrtx * cbrtx);
}

template <typename T, typename dT>
CUDA_HOST_DEVICE ValueAndPushforward<T, dT> hypot_pushforward(T x, T y, dT d_x,
                                                              dT d_y) {
  T h = ::std::hypot(x, y);
  return {h, (x * d_x + y * d_y) / h};
}

template <typename T, typename U>
CUDA_HOST_DEVICE void hypot_pullback(T x, T y, U d_z, T* d_x, T* d_y) {
  T h = ::std::hypot(x, y);
  *d_x += (x / h) * d_z;
  *d_y += (y / h) * d_z;
}

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

CUDA_HOST_DEVICE inline void sqrtf_pullback(float a, float d_y, float* d_a) {
  *d_a += (1.F / (2.F * sqrtf(a))) * d_y;
}

// These are required because C variants of mathematical functions are
// defined in global namespace.
// 1. Basic Math Functions
using std::abs_pullback;
using std::abs_pushforward;
using std::fabs_pullback;
using std::fabs_pushforward;
using std::fabsf_pullback;
using std::fabsf_pushforward;
using std::fabsl_pullback;
using std::fabsl_pushforward;
using std::fdim_pullback;
using std::fdim_pushforward;
using std::fdimf_pullback;
using std::fdimf_pushforward;
using std::fdiml_pullback;
using std::fdiml_pushforward;
using std::fma_pullback;
using std::fma_pushforward;
using std::fmaf_pullback;
using std::fmaf_pushforward;
using std::fmal_pullback;
using std::fmal_pushforward;
using std::fmax_pullback;
using std::fmax_pushforward;
using std::fmaxf_pullback;
using std::fmaxf_pushforward;
using std::fmaxl_pullback;
using std::fmaxl_pushforward;
using std::fmin_pullback;
using std::fmin_pushforward;
using std::fminf_pullback;
using std::fminf_pushforward;
using std::fminl_pullback;
using std::fminl_pushforward;
using std::fmod_pullback;
using std::fmod_pushforward;
using std::fmodf_pullback;
using std::fmodf_pushforward;
using std::fmodl_pullback;
using std::fmodl_pushforward;
using std::imaxabs_pullback;
using std::imaxabs_pushforward;
using std::llabs_pullback;
using std::llabs_pushforward;
using std::remainder_pullback;
using std::remainder_pushforward;
using std::remainderf_pullback;
using std::remainderf_pushforward;
using std::remainderl_pullback;
using std::remainderl_pushforward;

// 2. Exponential Functions
using std::exp2_pullback;
using std::exp2_pushforward;
using std::exp2f_pullback;
using std::exp2f_pushforward;
using std::exp2l_pullback;
using std::exp2l_pushforward;
using std::exp_pullback;
using std::exp_pushforward;
using std::expf_pullback;
using std::expf_pushforward;
using std::expl_pullback;
using std::expl_pushforward;
using std::expm1_pullback;
using std::expm1_pushforward;
using std::expm1f_pullback;
using std::expm1f_pushforward;
using std::expm1l_pullback;
using std::expm1l_pushforward;

// 3. Logarithmic Functions
using std::log10_pullback;
using std::log10_pushforward;
using std::log10f_pullback;
using std::log10f_pushforward;
using std::log10l_pullback;
using std::log10l_pushforward;
using std::log1p_pullback;
using std::log1p_pushforward;
using std::log1pf_pullback;
using std::log1pf_pushforward;
using std::log1pl_pullback;
using std::log1pl_pushforward;
using std::log2_pullback;
using std::log2_pushforward;
using std::log2f_pullback;
using std::log2f_pushforward;
using std::log2l_pullback;
using std::log2l_pushforward;
using std::log_pullback;
using std::log_pushforward;
using std::logf_pullback;
using std::logf_pushforward;
using std::logl_pullback;
using std::logl_pushforward;

// 4. Trigonometric Functions
using std::acos_pullback;
using std::acos_pushforward;
using std::acosf_pullback;
using std::acosf_pushforward;
using std::acosl_pullback;
using std::acosl_pushforward;
using std::asin_pullback;
using std::asin_pushforward;
using std::asinf_pullback;
using std::asinf_pushforward;
using std::asinl_pullback;
using std::asinl_pushforward;
using std::atan2_pullback;
using std::atan2_pushforward;
using std::atan2f_pullback;
using std::atan2f_pushforward;
using std::atan2l_pullback;
using std::atan2l_pushforward;
using std::atan_pullback;
using std::atan_pushforward;
using std::atanf_pullback;
using std::atanf_pushforward;
using std::atanl_pullback;
using std::atanl_pushforward;
using std::cos_pullback;
using std::cos_pushforward;
using std::cosf_pullback;
using std::cosf_pushforward;
using std::cosl_pullback;
using std::cosl_pushforward;
using std::sin_pullback;
using std::sin_pushforward;
using std::sinf_pullback;
using std::sinf_pushforward;
using std::sinl_pullback;
using std::sinl_pushforward;
using std::tan_pullback;
using std::tan_pushforward;
using std::tanf_pullback;
using std::tanf_pushforward;
using std::tanl_pullback;
using std::tanl_pushforward;

// 5. Hyperbolic Functions
using std::acosh_pullback;
using std::acosh_pushforward;
using std::acoshf_pullback;
using std::acoshf_pushforward;
using std::acoshl_pullback;
using std::acoshl_pushforward;
using std::asinh_pullback;
using std::asinh_pushforward;
using std::asinhf_pullback;
using std::asinhf_pushforward;
using std::asinhl_pullback;
using std::asinhl_pushforward;
using std::atanh_pullback;
using std::atanh_pushforward;
using std::atanhf_pullback;
using std::atanhf_pushforward;
using std::atanhl_pullback;
using std::atanhl_pushforward;
using std::cosh_pullback;
using std::cosh_pushforward;
using std::coshf_pullback;
using std::coshf_pushforward;
using std::coshl_pullback;
using std::coshl_pushforward;
using std::sinh_pullback;
using std::sinh_pushforward;
using std::sinhf_pullback;
using std::sinhf_pushforward;
using std::sinhl_pullback;
using std::sinhl_pushforward;
using std::tanh_pullback;
using std::tanh_pushforward;
using std::tanhf_pullback;
using std::tanhf_pushforward;
using std::tanhl_pullback;
using std::tanhl_pushforward;

// 6. Error Functions
using std::erf_pullback;
using std::erf_pushforward;
using std::erfc_pullback;
using std::erfc_pushforward;
using std::erfcf_pullback;
using std::erfcf_pushforward;
using std::erfcl_pullback;
using std::erfcl_pushforward;
using std::erff_pullback;
using std::erff_pushforward;
using std::erfl_pullback;
using std::erfl_pushforward;

using std::cbrt_pullback;
using std::cbrt_pushforward;
using std::ceil_pushforward;
using std::floor_pushforward;
using std::hypot_pullback;
using std::hypot_pushforward;
using std::max_pullback;
using std::max_pushforward;
using std::min_pullback;
using std::min_pushforward;
using std::pow_pullback;
using std::pow_pushforward;
using std::sqrt_pushforward;

namespace class_functions {
template <typename T, typename U>
void constructor_pullback(ValueAndPushforward<T, U> rhs,
                          ValueAndPushforward<T, U>* d_this,
                          ValueAndPushforward<T, U>* d_rhs) {
  d_rhs->pushforward += d_this->pushforward;
}
} // namespace class_functions
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
