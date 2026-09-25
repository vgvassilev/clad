#ifndef CLAD_DIFFERENTIATOR_BUILTINDERIVATIVESCUDA_CUH
#define CLAD_DIFFERENTIATOR_BUILTINDERIVATIVESCUDA_CUH

#include "clad/Differentiator/BuiltinDerivatives.h"
#include "clad/Differentiator/CladConfig.h"

namespace clad {

/// \ingroup rules
namespace custom_derivatives {

__device__ inline void __expf_pullback(float a, float d_y, float* d_a) {
  *d_a += expf(a) * d_y;
}
// A call to a unary function resolves to a pushforward in both directions: the
// pullback would only multiply by f'(a), which is what the pushforward already
// carries, so a pullback with no pushforward beside it can never be reached.
// Reaching one is an error rather than a fallback, and the error names the
// caller's own line and calls a derivative clad ships "user-defined".
__device__ inline ValueAndPushforward<float, float>
__expf_pushforward(float a, float d_a) {
  return {__expf(a), __expf(a) * d_a};
}

__device__ inline void __logf_pullback(float a, float d_y, float* d_a) {
  *d_a += (1.F / a) * d_y;
}
__device__ inline ValueAndPushforward<float, float>
__logf_pushforward(float a, float d_a) {
  return {__logf(a), (1.F / a) * d_a};
}

__device__ inline void __fdividef_pullback(float a, float b, float d_y,
                                           float* d_a, float* d_b) {
  *d_a += (1.F / b) * d_y;
  *d_b += (-a / (b * b)) * d_y;
}

__device__ inline void rsqrtf_pullback(float a, float d_y, float* d_a) {
  // Compute the gradient of rsqrt with respect to x
  *d_a = d_y * (-0.5 * powf(a, -1.5));
}
__device__ inline ValueAndPushforward<float, float>
rsqrtf_pushforward(float a, float d_a) {
  return {rsqrtf(a), -0.5F * d_a * powf(a, -1.5F)};
}

__device__ inline void make_float2_pullback(float a, float b, float2 d_y,
                                            float* d_a, float* d_b) {
  *d_a += d_y.x;
  *d_b += d_y.y;
}
} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_BUILTINDERIVATIVESCUDA_CUH
