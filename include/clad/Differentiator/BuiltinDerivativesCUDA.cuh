#include "clad/Differentiator/CladConfig.h"

namespace clad {

namespace custom_derivatives {

__device__ inline void __expf_pullback(float a, float d_y, float* d_a) {
  *d_a += expf(a) * d_y;
}

__device__ inline void __logf_pullback(float a, float d_y, float* d_a) {
  *d_a += (1.F / a) * d_y;
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

__device__ inline void make_float2_pullback(float a, float b, float2 d_y,
                                            float* d_a, float* d_b) {
  *d_a += d_y.x;
  *d_b += d_y.y;
}
} // namespace custom_derivatives
} // namespace clad
