#ifndef CLAD_DIFFERENTIATOR_BUILTINDERIVATIVESCUDA_CUH
#define CLAD_DIFFERENTIATOR_BUILTINDERIVATIVESCUDA_CUH

#include "clad/Differentiator/BuiltinDerivatives.h"
#include "clad/Differentiator/CladConfig.h"

namespace clad {

/// \ingroup rules
namespace custom_derivatives {

// These three are unary, and a call to a unary function resolves to a
// pushforward in both directions -- canUsePushforwardInRevMode -- because the
// pullback would only multiply by f'(a), which is the pushforward's second
// half already. Each used to carry a pullback and no pushforward, and nothing
// could reach it: the lookup fell through to a hard error naming the caller's
// own line and calling a derivative clad ships "user-defined".
//
// None of them carries a pullback now. Keeping one is not merely dead weight:
// with a pushforward beside it clad re-runs the pullback lookup to warn that
// the pullback goes unused, and that lookup wants two parameters where a real
// pullback has three, so the warning becomes the very error being fixed.__device__ inline ValueAndPushforward<float, float>
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
