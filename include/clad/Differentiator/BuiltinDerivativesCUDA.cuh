#ifndef CLAD_DIFFERENTIATOR_BUILTINDERIVATIVESCUDA_CUH
#define CLAD_DIFFERENTIATOR_BUILTINDERIVATIVESCUDA_CUH

#include "clad/Differentiator/BuiltinDerivatives.h"
#include "clad/Differentiator/CladConfig.h"

namespace clad {

/// \ingroup rules
namespace custom_derivatives {

// These three are unary, so a call to one resolves to a pushforward in both
// directions; a pullback would only multiply by f'(a), which is the
// pushforward's second half already. Do not add a pullback back: with a
// pushforward beside it, the unused-pullback lookup expects two parameters
// where a real pullback has three, and then fails with the very error this
// pushforward exists to avoid (#2172).
__device__ inline ValueAndPushforward<float, float>
__expf_pushforward(float a, float d_a) {
  return {__expf(a), __expf(a) * d_a};
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
