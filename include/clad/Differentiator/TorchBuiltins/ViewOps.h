#ifndef CLAD_DIFFERENTIATOR_TORCHBUILTINS_VIEWOPS_H
#define CLAD_DIFFERENTIATOR_TORCHBUILTINS_VIEWOPS_H

#include "clad/Differentiator/TorchBuiltins/TensorLifecycle.h" // IWYU pragma: export

#include <ATen/ops/permute.h>
#include <ATen/ops/transpose.h>

namespace clad::custom_derivatives::at {

inline void transpose_pullback(const ::at::Tensor& input, int64_t dim0,
                               int64_t dim1, ::at::Tensor d_output,
                               ::at::Tensor* d_input, int64_t* /*d_dim0*/,
                               int64_t* /*d_dim1*/) {
  if (!d_input || !::clad::torch::detail::should_propagate(d_output))
    return;
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_tensor(input);
  ::clad::torch::detail::accumulate(d_input,
                                    ::at::transpose(d_output, dim0, dim1));
}

inline void permute_pullback(const ::at::Tensor& input, ::at::IntArrayRef dims,
                             ::at::Tensor d_output, ::at::Tensor* d_input,
                             ::at::IntArrayRef* /*d_dims*/) {
  if (!d_input || !::clad::torch::detail::should_propagate(d_output))
    return;
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_tensor(input);
  auto inverse = ::clad::torch::detail::inverse_permutation(dims, input.dim());
  ::clad::torch::detail::accumulate(d_input, ::at::permute(d_output, inverse));
}

} // namespace clad::custom_derivatives::at

#endif // CLAD_DIFFERENTIATOR_TORCHBUILTINS_VIEWOPS_H
