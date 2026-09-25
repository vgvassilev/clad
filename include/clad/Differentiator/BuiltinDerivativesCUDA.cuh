#ifndef CLAD_DIFFERENTIATOR_BUILTINDERIVATIVESCUDA_CUH
#define CLAD_DIFFERENTIATOR_BUILTINDERIVATIVESCUDA_CUH

#include "clad/Differentiator/BuiltinDerivatives.h"
#include "clad/Differentiator/CladConfig.h"

namespace clad {

/// \ingroup rules
namespace custom_derivatives {

// Reverse mode asks a function of one argument for its pushforward rather
// than for its pullback: the reverse of a unary call is a multiplication by
// f'(x), so there is nothing left over for a separate pullback to do. A
// pullback for a unary function is therefore dead code, and reaching one is a
// hard error rather than a fallback.

__device__ inline ValueAndPushforward<float, float>
__expf_pushforward(float a, float d_a) {
  return {__expf(a), expf(a) * d_a};
}

__device__ inline void __expf_pullback(float a, float d_y, float* d_a) {
  *d_a += expf(a) * d_y;
}

__device__ inline ValueAndPushforward<float, float>
__logf_pushforward(float a, float d_a) {
  return {__logf(a), (1.F / a) * d_a};
}

__device__ inline void __logf_pullback(float a, float d_y, float* d_a) {
  *d_a += (1.F / a) * d_y;
}

__device__ inline void __fdividef_pullback(float a, float b, float d_y,
                                           float* d_a, float* d_b) {
  *d_a += (1.F / b) * d_y;
  *d_b += (-a / (b * b)) * d_y;
}

__device__ inline ValueAndPushforward<float, float>
rsqrtf_pushforward(float a, float d_a) {
  // Compute the gradient of rsqrt with respect to x
  return {rsqrtf(a), -0.5f * powf(a, -1.5f) * d_a};
}

__device__ inline void rsqrtf_pullback(float a, float d_y, float* d_a) {
  // Compute the gradient of rsqrt with respect to x
  *d_a = d_y * (-0.5 * powf(a, -1.5));
}

__device__ inline void make_float2_pullback(float a, float b, float2 d_y,
                                            float* d_a, float* d_b) {
  *d_a += d_y.x;
  *d_b += d_y.y;
}
} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_BUILTINDERIVATIVESCUDA_CUH
