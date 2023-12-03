#ifndef CLAD_DIFF_MODE_H
#define CLAD_DIFF_MODE_H

namespace clad {
enum class DiffMode {
  unknown = 0,
  forward,
  vector_forward_mode,
  experimental_pushforward,
  experimental_pullback,
  experimental_vector_pushforward,
  reverse,
  hessian,
  jacobian,
  reverse_mode_forward_pass,
  error_estimation
};
}

#endif
