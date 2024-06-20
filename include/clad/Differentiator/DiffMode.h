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
  hessian_diagonal,
  jacobian,
  reverse_mode_forward_pass,
  error_estimation
};

/// Convert enum value to string.
inline const char* DiffModeToString(DiffMode mode) {
  switch (mode) {
  case DiffMode::forward:
    return "forward";
  case DiffMode::vector_forward_mode:
    return "vector_forward_mode";
  case DiffMode::experimental_pushforward:
    return "pushforward";
  case DiffMode::experimental_pullback:
    return "pullback";
  case DiffMode::experimental_vector_pushforward:
    return "vector_pushforward";
  case DiffMode::reverse:
    return "reverse";
  case DiffMode::hessian:
    return "hessian";
  case DiffMode::hessian_diagonal:
    return "hessian_diagonal";
  case DiffMode::jacobian:
    return "jacobian";
  case DiffMode::reverse_mode_forward_pass:
    return "reverse_mode_forward_pass";
  case DiffMode::error_estimation:
    return "error_estimation";
  default:
    return "unknown";
  }
}
}

#endif
