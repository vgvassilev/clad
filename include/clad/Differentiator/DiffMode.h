#ifndef CLAD_DIFF_MODE_H
#define CLAD_DIFF_MODE_H

namespace clad {
  enum class DiffMode {
    unknown = 0,
    forward,
    experimental_pushforward,
    reverse,
    hessian,
    jacobian,
    error_estimation
  };
}

#endif