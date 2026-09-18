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
#include <cstdlib>
#include <cstring>
#include <functional>

#define elidable_reverse_forw __attribute__((annotate("elidable_reverse_forw")))

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

/// Empty payload for a pullback that carries no reverse-pass state.
struct no_state {};

/// Per-call state a custom `X_reverse_forw` hands to its matching `X_pullback`.
/// The reverse_forw takes a trailing `pullback_state<Payload>&` out-parameter
/// and fills it in the forward sweep; the pullback takes a trailing
/// `pullback_state<Payload>`, by value or by reference, and reads it in the
/// reverse sweep. Take it by reference when the payload owns storage that
/// cannot be copied, such as a `clad::tape<T>` the pullback pops from. clad
/// declares one carrier and threads it into both calls -- the same out-param
/// mechanism as `clad::restore_tracker`, and it composes with the
/// `ValueAndAdjoint` a value-returning reverse_forw already returns. Use it to
/// hand computed-in-forward data (e.g. a sort permutation) to the pullback
/// without a shared or global stash.
///
/// `Payload` is author-owned; extend it by appending fields. An empty
/// `pullback_state<no_state>` folds away at -O1+. When a reverse_forw also
/// takes a `restore_tracker&`, the `pullback_state<Payload>&` comes first.
template <typename Payload = no_state> struct pullback_state {
  Payload data{};
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
/// to determine the class. To solve this, 'clad::Tag<T>' is
/// used. A custom_derivative pushforward for constructor is required to have
/// 'clad::Tag<T>' as the first argument, where 'T' is the
/// class for which constructor pushforward is defined.
/// We do the same for constructor_reverse_forw.
template <class T> class Tag {};

/// Marks an entity non-differentiable: clad treats it as opaque and never
/// clones its body to synthesize a derivative. Apply at the declaration of a
/// variable, member, function, or type you own, e.g.
///   struct CLAD_NONDIFFERENTIABLE Handle { double* data; };
#define CLAD_NONDIFFERENTIABLE __attribute__((annotate("non_differentiable")))

/// Marks a type you do NOT own -- a library type you cannot annotate at its own
/// declaration -- non-differentiable, by specializing clad::Tag for it. Use at
/// global scope. The type may contain commas, e.g.
///   CLAD_NONDIFFERENTIABLE_TYPE(std::map<int, double>);
#define CLAD_NONDIFFERENTIABLE_TYPE(...)                                       \
  namespace clad {                                                             \
  template <> class CLAD_NONDIFFERENTIABLE Tag<__VA_ARGS__> {};                \
  }

/// We have aliases with for old tags for backwards compatibility.
template <class T> using ConstructorPushforwardTag = Tag<T>;

template <class T> using ConstructorReverseForwTag = Tag<T>;

namespace custom_derivatives {
#ifdef __CUDACC__
template <typename T>
ValueAndPushforward<cudaError_t, cudaError_t>
cudaMalloc_pushforward(T** devPtr, size_t sz, T** d_devPtr, size_t d_sz)
    __attribute__((host)) {
  return {cudaMalloc(devPtr, sz), cudaMalloc(d_devPtr, sz)};
}

ValueAndPushforward<cudaError_t, cudaError_t>
cudaMemcpy_pushforward(void* destPtr, const void* srcPtr, size_t count,
                       cudaMemcpyKind kind, void* d_destPtr,
                       const void* d_srcPtr, size_t d_count) {
  return {cudaMemcpy(destPtr, srcPtr, count, kind),
          cudaMemcpy(d_destPtr, d_srcPtr, count, kind)};
}

ValueAndPushforward<int, int> cudaDeviceSynchronize_pushforward() {
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

template <typename T>
cudaError_t cudaMalloc_reverse_forw(T** devPtr, size_t sz, T** d_devPtr,
                                    size_t d_sz)
    __attribute__((host)) elidable_reverse_forw;

template <typename T>
void cudaMalloc_pullback(T** devPtr, size_t sz, cudaError_t d_ret, T** d_devPtr,
                         size_t* d_sz) __attribute__((host));

cudaError_t cudaFree_reverse_forw(void* ptr, void* d_ptr) elidable_reverse_forw;

void cudaFree_pullback(void* ptr, cudaError_t d_ret, void* d_ptr);

template <typename... Args>
unsigned __cudaPushCallConfiguration_reverse_forw(Args...);
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


} // namespace custom_derivatives
} // namespace clad

#include "clad/Differentiator/LibCDerivatives.h"


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
